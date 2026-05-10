#include "bvh.cuh"
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <vector>
#include <cstdio>
#include <cuda_runtime.h>

namespace futaba {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK_BVH(call)                                                   \
    do {                                                                       \
        const cudaError_t _e = (call);                                         \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "[optix_bvh] CUDA error %s:%d  %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
        }                                                                      \
    } while (0)

#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        const OptixResult _r = (call);                                         \
        if (_r != OPTIX_SUCCESS) {                                             \
            fprintf(stderr, "[optix_bvh] OptiX error %s:%d  code=%d\n",       \
                    __FILE__, __LINE__, (int)_r);                               \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Global OptiX context (initialised once per process)
// ---------------------------------------------------------------------------

static OptixDeviceContext g_optixContext = nullptr;

OptixDeviceContext getOptixContext() {
    return g_optixContext;
}

void initOptix() {
    if (g_optixContext) return;

    // Ensure a CUDA context exists.
    CUDA_CHECK_BVH(cudaFree(nullptr));

    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions options = {};
#ifndef NDEBUG
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
    OPTIX_CHECK(optixDeviceContextCreate(nullptr, &options, &g_optixContext));
}

// ---------------------------------------------------------------------------
// BVH build
//
// Uses indexed triangle geometry so that:
//  • OptiX can apply per-primitive flags correctly.
//  • Vertex welding can be added later without changing the build path.
//  • The API accurately reflects the intended geometry layout.
//
// Note: vertices are currently NOT welded (one entry per triangle corner),
// so memory usage is the same as a flat layout.  Welding is a separate pass.
// ---------------------------------------------------------------------------

void buildOptixBVH(BVH& bvh, const Triangle* hostTriangles, uint32_t triangleCount) {
    if (triangleCount == 0) return;
    initOptix();

    // Build a flat vertex array (3 unique entries per triangle) and a matching
    // index buffer.  No deduplication yet; both are uploaded to device memory.
    const uint32_t vertexCount = triangleCount * 3;

    std::vector<float3> vertices(vertexCount);
    std::vector<uint3>  indices(triangleCount);

    for (uint32_t i = 0; i < triangleCount; ++i) {
        const uint32_t base = i * 3;
        vertices[base + 0] = make_float3(hostTriangles[i].p0.x,
                                          hostTriangles[i].p0.y,
                                          hostTriangles[i].p0.z);
        vertices[base + 1] = make_float3(hostTriangles[i].p1.x,
                                          hostTriangles[i].p1.y,
                                          hostTriangles[i].p1.z);
        vertices[base + 2] = make_float3(hostTriangles[i].p2.x,
                                          hostTriangles[i].p2.y,
                                          hostTriangles[i].p2.z);
        indices[i] = make_uint3(base, base + 1, base + 2);
    }

    // Upload vertex and index buffers to device.
    CUdeviceptr d_vertices = 0;
    CUdeviceptr d_indices  = 0;

    const size_t vertexBytes = vertexCount    * sizeof(float3);
    const size_t indexBytes  = triangleCount  * sizeof(uint3);

    CUDA_CHECK_BVH(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertexBytes));
    CUDA_CHECK_BVH(cudaMemcpy(reinterpret_cast<void*>(d_vertices),
                               vertices.data(), vertexBytes, cudaMemcpyHostToDevice));

    CUDA_CHECK_BVH(cudaMalloc(reinterpret_cast<void**>(&d_indices), indexBytes));
    CUDA_CHECK_BVH(cudaMemcpy(reinterpret_cast<void*>(d_indices),
                               indices.data(), indexBytes, cudaMemcpyHostToDevice));

    // Describe the triangle geometry to OptiX.
    unsigned int geometryFlags = OPTIX_GEOMETRY_FLAG_NONE;

    OptixBuildInput buildInput = {};
    buildInput.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes         = sizeof(float3);
    buildInput.triangleArray.numVertices                 = vertexCount;
    buildInput.triangleArray.vertexBuffers               = &d_vertices;
    buildInput.triangleArray.indexFormat                 = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes          = sizeof(uint3);
    buildInput.triangleArray.numIndexTriplets            = triangleCount;
    buildInput.triangleArray.indexBuffer                 = d_indices;
    buildInput.triangleArray.flags                       = &geometryFlags;
    buildInput.triangleArray.numSbtRecords               = 1;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
                              OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

    // Query memory requirements.
    OptixAccelBufferSizes bufferSizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        g_optixContext, &accelOptions, &buildInput, 1, &bufferSizes));

    CUdeviceptr d_tempBuffer = 0;
    CUDA_CHECK_BVH(cudaMalloc(reinterpret_cast<void**>(&d_tempBuffer),
                               bufferSizes.tempSizeInBytes));
    CUDA_CHECK_BVH(cudaMalloc(reinterpret_cast<void**>(&bvh.accelBuffer),
                               bufferSizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        g_optixContext,
        nullptr,                   // default stream
        &accelOptions,
        &buildInput,
        1,
        d_tempBuffer,              bufferSizes.tempSizeInBytes,
        bvh.accelBuffer,           bufferSizes.outputSizeInBytes,
        &bvh.traversable,
        nullptr, 0                 // emitted properties (none)
    ));

    CUDA_CHECK_BVH(cudaDeviceSynchronize());

    // Temporary buffers are no longer needed after the build.
    CUDA_CHECK_BVH(cudaFree(reinterpret_cast<void*>(d_tempBuffer)));
    CUDA_CHECK_BVH(cudaFree(reinterpret_cast<void*>(d_indices)));
    CUDA_CHECK_BVH(cudaFree(reinterpret_cast<void*>(d_vertices)));
}

void clearOptixBVH(BVH& bvh) {
    if (bvh.accelBuffer) {
        CUDA_CHECK_BVH(cudaFree(reinterpret_cast<void*>(bvh.accelBuffer)));
        bvh.accelBuffer = 0;
    }
    bvh.traversable = 0;
}

} // namespace futaba
