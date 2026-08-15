#include "optix_pipeline.h"
#include "renderer.h"
#include "optix_check.h"
#include "common.cuh"
#include <iostream>
#include <vector>
#include <optix_stubs.h>

FUTABA_NAMESPACE_BEGIN
    extern OptixDeviceContext getOptixContext();
    extern void initOptix();
FUTABA_NAMESPACE_END

namespace {

// Small helpers that turn the repetitive "fill a OptixProgramGroupDesc,
// call optixProgramGroupCreate" boilerplate into one line per program group.
OptixProgramGroup createRaygenPG(OptixDeviceContext context, OptixModule module,
                                 const char *entryFunctionName,
                                 const OptixProgramGroupOptions &pgOptions,
                                 char *log, size_t logCapacity) {
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc.raygen.module = module;
    desc.raygen.entryFunctionName = entryFunctionName;

    OptixProgramGroup pg;
    size_t sizeofLog = logCapacity;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &desc, 1, &pgOptions, log, &sizeofLog, &pg), log);
    return pg;
}

OptixProgramGroup createMissPG(OptixDeviceContext context, OptixModule module,
                               const char *entryFunctionName,
                               const OptixProgramGroupOptions &pgOptions,
                               char *log, size_t logCapacity) {
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    desc.miss.module = module;
    desc.miss.entryFunctionName = entryFunctionName;

    OptixProgramGroup pg;
    size_t sizeofLog = logCapacity;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &desc, 1, &pgOptions, log, &sizeofLog, &pg), log);
    return pg;
}

// `closestHitName` and/or `anyHitName` may be null to omit that shader.
OptixProgramGroup createHitgroupPG(OptixDeviceContext context, OptixModule module,
                                   const char *closestHitName, const char *anyHitName,
                                   const OptixProgramGroupOptions &pgOptions,
                                   char *log, size_t logCapacity) {
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    if (closestHitName) {
        desc.hitgroup.moduleCH = module;
        desc.hitgroup.entryFunctionNameCH = closestHitName;
    }
    if (anyHitName) {
        desc.hitgroup.moduleAH = module;
        desc.hitgroup.entryFunctionNameAH = anyHitName;
    }

    OptixProgramGroup pg;
    size_t sizeofLog = logCapacity;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &desc, 1, &pgOptions, log, &sizeofLog, &pg), log);
    return pg;
}

// Uploads a SBT record table to a fresh device allocation.
void *uploadSbtRecords(const std::vector<EmptyRecord> &records) {
    void *d_records = nullptr;
    const size_t bytes = records.size() * sizeof(EmptyRecord);
    CUDA_CHECK(cudaMalloc(&d_records, bytes));
    CUDA_CHECK(cudaMemcpy(d_records, records.data(), bytes, cudaMemcpyHostToDevice));
    return d_records;
}

} // namespace

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
    futaba::g_optixCompileStatus = "Initializing OptiX & CUDA context...";
    futaba::initOptix();
    OptixDeviceContext context = futaba::getOptixContext();

    // Pipeline compile options
    futaba::g_optixCompileProgress = 0.12f;
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
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    // Load PTX bytecode
    futaba::g_optixCompileProgress = 0.18f;
    futaba::g_optixCompileStatus = "Loading PTX bytecode...";
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

    futaba::g_optixCompileProgress = 0.25f;
    futaba::g_optixCompileStatus = "Compiling OptiX PTX device module (RT Cores)...";
    OPTIX_CHECK_LOG(optixModuleCreate(context, &moduleCompileOptions, &pipelineCompileOptions,
                                      ptxCode.data(), ptxSize, log, &sizeof_log, &module),
                    log);

    // Program groups
    OptixProgramGroupOptions pgOptions = {};

    futaba::g_optixCompileProgress = 0.50f;
    futaba::g_optixCompileStatus = "Creating ray generation & hit programs...";
    OptixProgramGroup raygenProgGroupRender =
        createRaygenPG(context, module, "__raygen__render", pgOptions, log, sizeof(log));

    OptixProgramGroup raygenProgGroupPath =
        createRaygenPG(context, module, "__raygen__path", pgOptions, log, sizeof(log));

    OptixProgramGroup missProgGroup =
        createMissPG(context, module, "__miss__ms", pgOptions, log, sizeof(log));

    OptixProgramGroup hitProgGroup =
        createHitgroupPG(context, module, "__closesthit__ch", nullptr, pgOptions, log, sizeof(log));

    // Shadow miss & hit
    futaba::g_optixCompileProgress = 0.70f;
    futaba::g_optixCompileStatus = "Creating shadow & miss shader programs...";
    OptixProgramGroup shadowMissProgGroup =
        createMissPG(context, module, "__miss__shadow", pgOptions, log, sizeof(log));

    OptixProgramGroup shadowHitProgGroup =
        createHitgroupPG(context, module, nullptr, "__anyhit__shadow", pgOptions, log, sizeof(log));

    // Create OptiX pipeline
    futaba::g_optixCompileProgress = 0.85f;
    futaba::g_optixCompileStatus = "Linking OptiX rendering pipeline...";
    OptixProgramGroup programGroups[] = {raygenProgGroupRender,
                                         raygenProgGroupPath,
                                         missProgGroup,
                                         hitProgGroup,
                                         shadowMissProgGroup,
                                         shadowHitProgGroup};
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 1;

    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(context, &pipelineCompileOptions, &pipelineLinkOptions,
                                        programGroups, 6, log, &sizeof_log, &pipeline),
                    log);

    // Build Shader Binding Table (SBT)
    futaba::g_optixCompileProgress = 0.95f;
    futaba::g_optixCompileStatus = "Building Shader Binding Table (SBT)...";

    // Raygen Records
    std::vector<EmptyRecord> raygenRecords(2);
    optixSbtRecordPackHeader(raygenProgGroupRender, &raygenRecords[0]);
    optixSbtRecordPackHeader(raygenProgGroupPath, &raygenRecords[1]);
    d_raygenRecordsBase.reset(reinterpret_cast<char *>(uploadSbtRecords(raygenRecords)));
    raygenRecordRender = reinterpret_cast<CUdeviceptr>(d_raygenRecordsBase.get());
    raygenRecordPath   = reinterpret_cast<CUdeviceptr>(d_raygenRecordsBase.get() + sizeof(EmptyRecord));
    sbt.raygenRecord   = raygenRecordRender;

    // Miss Records
    std::vector<EmptyRecord> missRecords(2);
    optixSbtRecordPackHeader(missProgGroup, &missRecords[0]);
    optixSbtRecordPackHeader(shadowMissProgGroup, &missRecords[1]);
    d_missRecordBase.reset(reinterpret_cast<char *>(uploadSbtRecords(missRecords)));
    sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(d_missRecordBase.get());
    sbt.missRecordStrideInBytes = sizeof(EmptyRecord);
    sbt.missRecordCount = 2;

    // Hit Records
    std::vector<EmptyRecord> hitRecords(2);
    optixSbtRecordPackHeader(hitProgGroup, &hitRecords[0]);
    optixSbtRecordPackHeader(shadowHitProgGroup, &hitRecords[1]);
    d_hitRecordBase.reset(reinterpret_cast<char *>(uploadSbtRecords(hitRecords)));
    sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(d_hitRecordBase.get());
    sbt.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);
    sbt.hitgroupRecordCount = 2;

    // Parameter Buffer
    futaba::LaunchParams* d_p = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_p), sizeof(futaba::LaunchParams)));
    d_params.reset(d_p);

    // Destroy local program groups as they are linked into pipeline
    optixProgramGroupDestroy(raygenProgGroupRender);
    optixProgramGroupDestroy(raygenProgGroupPath);
    optixProgramGroupDestroy(missProgGroup);
    optixProgramGroupDestroy(hitProgGroup);
    optixProgramGroupDestroy(shadowMissProgGroup);
    optixProgramGroupDestroy(shadowHitProgGroup);

    futaba::g_optixCompileProgress = 1.00f;
    futaba::g_optixCompileStatus = "Initialization complete";
    futaba::g_optixCompileCompleted = true;
}

FUTABA_NAMESPACE_BEGIN
    void cleanup_pipeline() {
        g_pipeline.cleanup();
    }
FUTABA_NAMESPACE_END
