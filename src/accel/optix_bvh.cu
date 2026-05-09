#include "bvh.cuh"
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

namespace futaba {

OptixDeviceContext g_optixContext = nullptr;

OptixDeviceContext getOptixContext() {
    return g_optixContext;
}

void initOptix() {
    if (g_optixContext) return;
    cudaFree(0); // Initialize CUDA context if not already done
    optixInit();
    OptixDeviceContextOptions options = {};
    optixDeviceContextCreate(0, &options, &g_optixContext);
}

void buildOptixBVH(BVH& bvh, const Triangle* hostTriangles, uint32_t triangleCount) {
    if (triangleCount == 0) return;
    initOptix();

    std::vector<float3> vertices(triangleCount * 3);
    for (uint32_t i = 0; i < triangleCount; ++i) {
        vertices[i * 3 + 0] = make_float3(hostTriangles[i].p0.x, hostTriangles[i].p0.y, hostTriangles[i].p0.z);
        vertices[i * 3 + 1] = make_float3(hostTriangles[i].p1.x, hostTriangles[i].p1.y, hostTriangles[i].p1.z);
        vertices[i * 3 + 2] = make_float3(hostTriangles[i].p2.x, hostTriangles[i].p2.y, hostTriangles[i].p2.z);
    }

    CUdeviceptr d_vertices;
    size_t verticesSize = vertices.size() * sizeof(float3);
    cudaMalloc(reinterpret_cast<void**>(&d_vertices), verticesSize);
    cudaMemcpy(reinterpret_cast<void*>(d_vertices), vertices.data(), verticesSize, cudaMemcpyHostToDevice);

    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.numVertices = triangleCount * 3;
    buildInput.triangleArray.vertexBuffers = &d_vertices;

    unsigned int buildFlags = OPTIX_GEOMETRY_FLAG_NONE;
    buildInput.triangleArray.flags = &buildFlags;
    buildInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes bufferSizes;
    optixAccelComputeMemoryUsage(g_optixContext, &accelOptions, &buildInput, 1, &bufferSizes);

    CUdeviceptr d_tempBuffer;
    cudaMalloc(reinterpret_cast<void**>(&d_tempBuffer), bufferSizes.tempSizeInBytes);
    cudaMalloc(reinterpret_cast<void**>(&bvh.accelBuffer), bufferSizes.outputSizeInBytes);

    optixAccelBuild(
        g_optixContext,
        0, // stream
        &accelOptions,
        &buildInput,
        1,
        d_tempBuffer,
        bufferSizes.tempSizeInBytes,
        bvh.accelBuffer,
        bufferSizes.outputSizeInBytes,
        &bvh.traversable,
        nullptr, // emitted properties
        0        // num emitted properties
    );

    cudaDeviceSynchronize();

    cudaFree(reinterpret_cast<void*>(d_tempBuffer));
    cudaFree(reinterpret_cast<void*>(d_vertices));
}

void clearOptixBVH(BVH& bvh) {
    if (bvh.accelBuffer) {
        cudaFree(reinterpret_cast<void*>(bvh.accelBuffer));
        bvh.accelBuffer = 0;
    }
    bvh.traversable = 0;
}

} // namespace futaba
