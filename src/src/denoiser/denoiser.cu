#include "denoiser.h"
#include "common.cuh"
#include "tonemapping.cuh"
#include <optix_stubs.h>
#include <iostream>

namespace futaba {

// Preparation kernel to divide accumulated color/albedo/normal values by sampleCount
__global__ void prepare_denoiser_input_kernel(
    const Color3f* film_pixels,
    const Color3f* albedo_buffer,
    const Color3f* normal_buffer,
    int width, int height, int sampleCount,
    float4* input_beauty,
    float4* input_albedo,
    float4* input_normal)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float inv_s = 1.0f / (float)sampleCount;

    // 1. Beauty (RGB)
    Color3f b = film_pixels[idx] * inv_s;
    input_beauty[idx] = make_float4(b.x, b.y, b.z, 1.0f);

    // 2. Albedo (RGB)
    Color3f a = albedo_buffer[idx] * inv_s;
    input_albedo[idx] = make_float4(a.x, a.y, a.z, 1.0f);

    // 3. Normal (XYZ in world -> camera space, and normalized)
    Color3f n = safe_normalize(normal_buffer[idx] * inv_s);
    input_normal[idx] = make_float4(n.x, n.y, n.z, 0.0f);
}

// Post-processing kernel to tonemap the denoised beauty and copy it to the PBO
__global__ void tonemap_and_copy_kernel(
    const float4* denoised_beauty,
    int width, int height,
    int tonemapping_mode,
    uchar4* pbo_ptr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    float4 beauty = denoised_beauty[idx];
    Color3f linear_avg(beauty.x, beauty.y, beauty.z);

    // Apply tonemapping
    Color3f tonemapped = tonemap::apply(linear_avg, tonemapping_mode);
    
    // Apply sRGB gamma correction
    Color3f final_color = toSRGB(tonemapped);

    // Clamp and copy to PBO uchar4
    pbo_ptr[idx].x = (unsigned char)clamp(final_color.x * 255.f, 0.f, 255.f);
    pbo_ptr[idx].y = (unsigned char)clamp(final_color.y * 255.f, 0.f, 255.f);
    pbo_ptr[idx].z = (unsigned char)clamp(final_color.z * 255.f, 0.f, 255.f);
    pbo_ptr[idx].w = 255;
}


DenoiserManager::DenoiserManager()
    : m_context(nullptr), m_denoiser(nullptr), m_width(0), m_height(0),
      m_stateSize(0), m_scratchSize(0), m_dState(0), m_dScratch(0), m_dIntensity(0),
      m_dAlbedoBuffer(nullptr), m_dNormalBuffer(nullptr),
      m_dInputBeauty(nullptr), m_dInputAlbedo(nullptr), m_dInputNormal(nullptr), m_dOutputBeauty(nullptr)
{}

DenoiserManager::~DenoiserManager() {
    destroy();
    if (m_denoiser) {
        optixDenoiserDestroy(m_denoiser);
        m_denoiser = nullptr;
    }
}

bool DenoiserManager::init(OptixDeviceContext context, int width, int height) {
    m_context = context;
    m_width = width;
    m_height = height;

    OptixDenoiserOptions options = {};
    options.guideAlbedo = 1;
    options.guideNormal = 1;
    options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;

    OptixResult res = optixDenoiserCreate(m_context, OPTIX_DENOISER_MODEL_KIND_HDR, &options, &m_denoiser);
    if (res != OPTIX_SUCCESS) {
        std::cerr << "[Denoiser] Failed to create OptiX Denoiser: " << res << std::endl;
        return false;
    }

    resize(width, height);
    return true;
}

void DenoiserManager::resize(int width, int height) {
    m_width = width;
    m_height = height;

    // Free existing GPU buffers first
    destroy();

    // Query OptiX for state & scratch size requirements
    OptixDenoiserSizes sizes = {};
    optixDenoiserComputeMemoryResources(m_denoiser, m_width, m_height, &sizes);

    m_stateSize = sizes.stateSizeInBytes;
    m_scratchSize = sizes.withoutOverlapScratchSizeInBytes;

    cudaMalloc(reinterpret_cast<void **>(&m_dState), m_stateSize);
    cudaMalloc(reinterpret_cast<void **>(&m_dScratch), m_scratchSize);
    cudaMalloc(reinterpret_cast<void **>(&m_dIntensity), sizeof(float));

    // Allocate accumulation guide buffers for Albedo and Normals
    size_t count = (size_t)width * height;
    cudaMalloc(reinterpret_cast<void **>(&m_dAlbedoBuffer), count * sizeof(Color3f));
    cudaMalloc(reinterpret_cast<void **>(&m_dNormalBuffer), count * sizeof(Color3f));
    cudaMemset(m_dAlbedoBuffer, 0, count * sizeof(Color3f));
    cudaMemset(m_dNormalBuffer, 0, count * sizeof(Color3f));

    // Allocate intermediate float4 buffers for the OptiX pipeline
    cudaMalloc(reinterpret_cast<void **>(&m_dInputBeauty), count * sizeof(float4));
    cudaMalloc(reinterpret_cast<void **>(&m_dInputAlbedo), count * sizeof(float4));
    cudaMalloc(reinterpret_cast<void **>(&m_dInputNormal), count * sizeof(float4));
    cudaMalloc(reinterpret_cast<void **>(&m_dOutputBeauty), count * sizeof(float4));

    // Setup the OptiX denoiser state
    optixDenoiserSetup(
        m_denoiser,
        0, // CUDA stream
        m_width, m_height,
        m_dState, m_stateSize,
        m_dScratch, m_scratchSize
    );
}

void DenoiserManager::exec(Color3f* d_film_pixels, 
                          Color3f* d_albedo_buffer, 
                          Color3f* d_normal_buffer, 
                          int sampleCount, 
                          uchar4* d_pbo_ptr, 
                          int tonemapping_mode) 
{
    if (sampleCount <= 0) return;

    // 1. Prepare denoiser inputs
    dim3 block(16, 16);
    dim3 grid((m_width + block.x - 1) / block.x, (m_height + block.y - 1) / block.y);

    prepare_denoiser_input_kernel<<<grid, block>>>(
        d_film_pixels,
        d_albedo_buffer,
        d_normal_buffer,
        m_width, m_height, sampleCount,
        m_dInputBeauty,
        m_dInputAlbedo,
        m_dInputNormal
    );

    // 2. Setup OptiX image layer descriptions
    OptixImage2D inputBeauty = {};
    inputBeauty.data = reinterpret_cast<CUdeviceptr>(m_dInputBeauty);
    inputBeauty.width = m_width;
    inputBeauty.height = m_height;
    inputBeauty.rowStrideInBytes = m_width * sizeof(float4);
    inputBeauty.pixelStrideInBytes = sizeof(float4);
    inputBeauty.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    OptixImage2D outputBeauty = {};
    outputBeauty.data = reinterpret_cast<CUdeviceptr>(m_dOutputBeauty);
    outputBeauty.width = m_width;
    outputBeauty.height = m_height;
    outputBeauty.rowStrideInBytes = m_width * sizeof(float4);
    outputBeauty.pixelStrideInBytes = sizeof(float4);
    outputBeauty.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    OptixImage2D inputAlbedo = {};
    inputAlbedo.data = reinterpret_cast<CUdeviceptr>(m_dInputAlbedo);
    inputAlbedo.width = m_width;
    inputAlbedo.height = m_height;
    inputAlbedo.rowStrideInBytes = m_width * sizeof(float4);
    inputAlbedo.pixelStrideInBytes = sizeof(float4);
    inputAlbedo.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    OptixImage2D inputNormal = {};
    inputNormal.data = reinterpret_cast<CUdeviceptr>(m_dInputNormal);
    inputNormal.width = m_width;
    inputNormal.height = m_height;
    inputNormal.rowStrideInBytes = m_width * sizeof(float4);
    inputNormal.pixelStrideInBytes = sizeof(float4);
    inputNormal.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // Assemble layers
    OptixDenoiserGuideLayer guideLayer = {};
    guideLayer.albedo = inputAlbedo;
    guideLayer.normal = inputNormal;

    OptixDenoiserLayer layer = {};
    layer.input = inputBeauty;
    layer.output = outputBeauty;

    OptixDenoiserParams params = {};
    params.hdrIntensity = m_dIntensity;
    params.blendFactor = 0.0f;

    // Compute average logarithmic intensity (for HDR autoexposure calculation)
    optixDenoiserComputeIntensity(
        m_denoiser,
        0, // CUDA stream
        &inputBeauty,
        m_dIntensity,
        m_dScratch,
        m_scratchSize
    );

    // Invoke the OptiX AI Denoiser
    optixDenoiserInvoke(
        m_denoiser,
        0, // CUDA stream
        &params,
        m_dState, m_stateSize,
        &guideLayer,
        &layer,
        1, // numLayers
        0, 0, // offsets
        m_dScratch, m_scratchSize
    );

    // 3. Post-process (tonemap, sRGB, copy to display PBO)
    tonemap_and_copy_kernel<<<grid, block>>>(
        m_dOutputBeauty,
        m_width, m_height,
        tonemapping_mode,
        d_pbo_ptr
    );
}

void DenoiserManager::destroy() {
    if (m_dState) { cudaFree(reinterpret_cast<void*>(m_dState)); m_dState = 0; }
    if (m_dScratch) { cudaFree(reinterpret_cast<void*>(m_dScratch)); m_dScratch = 0; }
    if (m_dIntensity) { cudaFree(reinterpret_cast<void*>(m_dIntensity)); m_dIntensity = 0; }
    
    if (m_dAlbedoBuffer) { cudaFree(m_dAlbedoBuffer); m_dAlbedoBuffer = nullptr; }
    if (m_dNormalBuffer) { cudaFree(m_dNormalBuffer); m_dNormalBuffer = nullptr; }

    if (m_dInputBeauty) { cudaFree(m_dInputBeauty); m_dInputBeauty = nullptr; }
    if (m_dInputAlbedo) { cudaFree(m_dInputAlbedo); m_dInputAlbedo = nullptr; }
    if (m_dInputNormal) { cudaFree(m_dInputNormal); m_dInputNormal = nullptr; }
    if (m_dOutputBeauty) { cudaFree(m_dOutputBeauty); m_dOutputBeauty = nullptr; }
}

} // namespace futaba
