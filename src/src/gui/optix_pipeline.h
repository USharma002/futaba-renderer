#pragma once
#include <optix.h>
#include <cuda_runtime.h>
#include "launch_params.h"
#include "cuda_unique_ptr.h"

struct EmptyRecord {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

class OptixPipelineManager {
public:
    OptixPipeline pipeline = nullptr;
    OptixShaderBindingTable sbt = {};
    OptixModule module = nullptr;
    cudaStream_t renderStream = nullptr;

    futaba::CudaUniquePtr<futaba::LaunchParams> d_params;
    futaba::CudaUniquePtr<char> d_raygenRecordsBase;
    futaba::CudaUniquePtr<char> d_missRecordBase;
    futaba::CudaUniquePtr<char> d_hitRecordBase;

    OptixPipelineManager() = default;
    ~OptixPipelineManager();

    void init();
    void cleanup();
};

extern OptixPipelineManager g_pipeline;
