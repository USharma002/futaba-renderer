#include "optix_pipeline.h"
#include "renderer.h"
#include <iostream>
#include <vector>
#include <optix_stubs.h>

namespace futaba {
    extern OptixDeviceContext getOptixContext();
    extern void initOptix();
}

#define OPTIX_CHECK_LOG(call, logBuffer)                                       \
    do {                                                                       \
        OptixResult _res = (call);                                             \
        if (_res != OPTIX_SUCCESS) {                                           \
            std::cerr << "OptiX call (" #call ") failed with error code: "     \
                      << _res << std::endl;                                    \
            std::cerr << "OptiX Log:\n" << (logBuffer) << std::endl;           \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

OptixPipelineManager g_pipeline;

OptixPipelineManager::~OptixPipelineManager() {
    cleanup();
}

void OptixPipelineManager::cleanup() {
    if (pipeline) {
        optixPipelineDestroy(pipeline);
        pipeline = nullptr;
    }
    if (module) {
        optixModuleDestroy(module);
        module = nullptr;
    }
    if (renderStream) {
        cudaStreamDestroy(renderStream);
        renderStream = nullptr;
    }
    d_params.reset();
    d_raygenRecordsBase.reset();
    d_missRecordBase.reset();
    d_hitRecordBase.reset();
}

void OptixPipelineManager::init() {
    if (pipeline)
        return;
    
    // Create stream first
    if (!renderStream) {
        cudaStreamCreate(&renderStream);
    }
    
    futaba::g_optixCompileProgress = 0.05f;
    futaba::g_optixCompileStatus = "Initializing OptiX context...";
    futaba::initOptix();
    OptixDeviceContext context = futaba::getOptixContext();

    // 1. Pipeline Compile Options
    futaba::g_optixCompileProgress = 0.10f;
    futaba::g_optixCompileStatus = "Configuring pipeline options...";
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues = 2; // Pointer packed into 2 uints
    pipelineCompileOptions.numAttributeValues = 2; // Barycentrics (u, v)
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params_buffer";

    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = 50;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    // 2. Load PTX
    futaba::g_optixCompileProgress = 0.15f;
    futaba::g_optixCompileStatus = "Loading PTX code...";
    FILE *fp = fopen(PTX_FILE_PATH, "rb");
    if (!fp) {
      std::cerr << "Failed to open PTX file: " << PTX_FILE_PATH << std::endl;
      exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t ptxSize = ftell(fp);
    rewind(fp);
    std::vector<char> ptxCode(ptxSize + 1);
    fread(ptxCode.data(), 1, ptxSize, fp);
    fclose(fp);

    char log[8192];
    size_t sizeof_log = sizeof(log);

    futaba::g_optixCompileProgress = 0.20f;
    futaba::g_optixCompileStatus = "Compiling OptiX device module...";
    OPTIX_CHECK_LOG(optixModuleCreate(context, &moduleCompileOptions, &pipelineCompileOptions,
                                      ptxCode.data(), ptxSize, log, &sizeof_log, &module),
                    log);

    // 3. Program Groups
    OptixProgramGroupOptions pgOptions = {};

    futaba::g_optixCompileProgress = 0.35f;
    futaba::g_optixCompileStatus = "Creating shader programs (1/7)...";
    OptixProgramGroup raygenProgGroupRender;
    OptixProgramGroupDesc raygenDescRender = {};
    raygenDescRender.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDescRender.raygen.module = module;
    raygenDescRender.raygen.entryFunctionName = "__raygen__render";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &raygenDescRender, 1, &pgOptions, log,
                                            &sizeof_log, &raygenProgGroupRender),
                    log);

    futaba::g_optixCompileProgress = 0.40f;
    futaba::g_optixCompileStatus = "Creating shader programs (2/7)...";
    OptixProgramGroup raygenProgGroupPath;
    OptixProgramGroupDesc raygenDescPath = {};
    raygenDescPath.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDescPath.raygen.module = module;
    raygenDescPath.raygen.entryFunctionName = "__raygen__path";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &raygenDescPath, 1, &pgOptions, log,
                                            &sizeof_log, &raygenProgGroupPath),
                    log);

    futaba::g_optixCompileProgress = 0.45f;
    futaba::g_optixCompileStatus = "Creating shader programs (3/7)...";
    OptixProgramGroup raygenProgGroupVolPath;
    OptixProgramGroupDesc raygenDescVolPath = {};
    raygenDescVolPath.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDescVolPath.raygen.module = module;
    raygenDescVolPath.raygen.entryFunctionName = "__raygen__volpath";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &raygenDescVolPath, 1, &pgOptions, log,
                                            &sizeof_log, &raygenProgGroupVolPath),
                    log);

    futaba::g_optixCompileProgress = 0.50f;
    futaba::g_optixCompileStatus = "Creating shader programs (4/7)...";
    OptixProgramGroup missProgGroup;
    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = module;
    missDesc.miss.entryFunctionName = "__miss__ms";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &missDesc, 1, &pgOptions, log, &sizeof_log,
                                            &missProgGroup),
                    log);

    futaba::g_optixCompileProgress = 0.55f;
    futaba::g_optixCompileStatus = "Creating shader programs (5/7)...";
    OptixProgramGroup hitProgGroup;
    OptixProgramGroupDesc hitDesc = {};
    hitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitDesc.hitgroup.moduleCH = module;
    hitDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &hitDesc, 1, &pgOptions, log, &sizeof_log,
                                            &hitProgGroup),
                    log);

    // Shadow miss
    futaba::g_optixCompileProgress = 0.60f;
    futaba::g_optixCompileStatus = "Creating shader programs (6/7)...";
    OptixProgramGroup shadowMissProgGroup;
    OptixProgramGroupDesc shadowMissDesc = {};
    shadowMissDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    shadowMissDesc.miss.module = module;
    shadowMissDesc.miss.entryFunctionName = "__miss__shadow";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &shadowMissDesc, 1, &pgOptions, log,
                                            &sizeof_log, &shadowMissProgGroup),
                    log);

    // Shadow hit (anyhit only, no closest-hit)
    futaba::g_optixCompileProgress = 0.65f;
    futaba::g_optixCompileStatus = "Creating shader programs (7/7)...";
    OptixProgramGroup shadowHitProgGroup;
    OptixProgramGroupDesc shadowHitDesc = {};
    shadowHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    shadowHitDesc.hitgroup.moduleAH = module;
    shadowHitDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &shadowHitDesc, 1, &pgOptions, log,
                                            &sizeof_log, &shadowHitProgGroup),
                    log);

    // 4. Create Pipeline
    futaba::g_optixCompileProgress = 0.80f;
    futaba::g_optixCompileStatus = "Linking pipeline...";
    OptixProgramGroup programGroups[] = {raygenProgGroupRender,
                                         raygenProgGroupPath,
                                         raygenProgGroupVolPath,
                                         missProgGroup,
                                         hitProgGroup,
                                         shadowMissProgGroup,
                                         shadowHitProgGroup};
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 1;

    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(context, &pipelineCompileOptions, &pipelineLinkOptions,
                                        programGroups, 7, log, &sizeof_log, &pipeline),
                    log);

    // 5. Build SBT
    futaba::g_optixCompileProgress = 0.90f;
    futaba::g_optixCompileStatus = "Building Shader Binding Table (SBT)...";
    
    // Raygen Records
    std::vector<EmptyRecord> raygenRecords(3);
    optixSbtRecordPackHeader(raygenProgGroupRender, &raygenRecords[0]);
    optixSbtRecordPackHeader(raygenProgGroupPath, &raygenRecords[1]);
    optixSbtRecordPackHeader(raygenProgGroupVolPath, &raygenRecords[2]);
    
    char* d_raygen = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_raygen), 3 * sizeof(EmptyRecord));
    cudaMemcpy(reinterpret_cast<void *>(d_raygen), raygenRecords.data(),
               3 * sizeof(EmptyRecord), cudaMemcpyHostToDevice);
    d_raygenRecordsBase.reset(d_raygen);
    sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(d_raygenRecordsBase.get());

    // Miss Records
    std::vector<EmptyRecord> missRecords(2);
    optixSbtRecordPackHeader(missProgGroup, &missRecords[0]);
    optixSbtRecordPackHeader(shadowMissProgGroup, &missRecords[1]);
    
    char* d_miss = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_miss), 2 * sizeof(EmptyRecord));
    cudaMemcpy(reinterpret_cast<void *>(d_miss), missRecords.data(),
               2 * sizeof(EmptyRecord), cudaMemcpyHostToDevice);
    d_missRecordBase.reset(d_miss);
    sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(d_missRecordBase.get());
    sbt.missRecordStrideInBytes = sizeof(EmptyRecord);
    sbt.missRecordCount = 2;

    // Hit Records
    std::vector<EmptyRecord> hitRecords(2);
    optixSbtRecordPackHeader(hitProgGroup, &hitRecords[0]);
    optixSbtRecordPackHeader(shadowHitProgGroup, &hitRecords[1]);
    
    char* d_hit = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_hit), 2 * sizeof(EmptyRecord));
    cudaMemcpy(reinterpret_cast<void *>(d_hit), hitRecords.data(),
               2 * sizeof(EmptyRecord), cudaMemcpyHostToDevice);
    d_hitRecordBase.reset(d_hit);
    sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(d_hitRecordBase.get());
    sbt.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);
    sbt.hitgroupRecordCount = 2;

    // Parameter Buffer
    futaba::LaunchParams* d_p = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_p), sizeof(futaba::LaunchParams));
    d_params.reset(d_p);

    // Destroy local program groups as they are linked into pipeline
    optixProgramGroupDestroy(raygenProgGroupRender);
    optixProgramGroupDestroy(raygenProgGroupPath);
    optixProgramGroupDestroy(raygenProgGroupVolPath);
    optixProgramGroupDestroy(missProgGroup);
    optixProgramGroupDestroy(hitProgGroup);
    optixProgramGroupDestroy(shadowMissProgGroup);
    optixProgramGroupDestroy(shadowHitProgGroup);

    futaba::g_optixCompileProgress = 1.00f;
    futaba::g_optixCompileStatus = "Completed";
    futaba::g_optixCompileCompleted = true;
}

namespace futaba {
    void cleanup_pipeline() {
        g_pipeline.cleanup();
    }
}
