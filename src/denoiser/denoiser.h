#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include "types.cuh"

namespace futaba {

extern void initOptix();
extern OptixDeviceContext getOptixContext();

class DenoiserManager {
public:
    DenoiserManager();
    ~DenoiserManager();

    // Initialize the denoiser for a given OptiX context and dimensions
    bool init(OptixDeviceContext context, int width, int height);

    // Resize internal buffers when resolution changes
    void resize(int width, int height);

    // Execute denoising: averages buffers, invokes OptiX denoiser, tonemaps, and outputs to OpenGL PBO
    void exec(Color3f* d_film_pixels, 
              Color3f* d_albedo_buffer, 
              Color3f* d_normal_buffer, 
              int sampleCount, 
              uchar4* d_pbo_ptr, 
              int tonemapping_mode);

    // Free all allocated OptiX and CUDA memory
    void destroy();

    // Getters for guide buffers (allocated on GPU inside the denoiser manager)
    Color3f* getAlbedoBuffer() const { return m_dAlbedoBuffer; }
    Color3f* getNormalBuffer() const { return m_dNormalBuffer; }
    float4* getOutputBeauty() const { return m_dOutputBeauty; }

private:
    OptixDeviceContext m_context;
    OptixDenoiser m_denoiser;
    
    int m_width;
    int m_height;

    // OptiX denoiser memory resources
    size_t m_stateSize;
    size_t m_scratchSize;
    CUdeviceptr m_dState;
    CUdeviceptr m_dScratch;
    CUdeviceptr m_dIntensity; // OptiX HDR intensity computation buffer

    // GPU guide buffers (we allocate them here to keep HDRFilm clean)
    Color3f* m_dAlbedoBuffer;
    Color3f* m_dNormalBuffer;

    // Temporary/intermediate buffers for denoiser input/output
    float4* m_dInputBeauty;   // float3 or float4 beauty buffer
    float4* m_dInputAlbedo;   // float3 or float4 guide albedo
    float4* m_dInputNormal;   // float3 or float4 guide normal
    float4* m_dOutputBeauty;  // float3 or float4 denoised beauty
};

} // namespace futaba
