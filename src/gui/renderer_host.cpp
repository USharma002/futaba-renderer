#include "launch_params.h"
#include "renderer.h"
#include <iostream>
#include <optix.h>
#include <optix_stubs.h>
#include <vector>

using namespace futaba;

// Geometry helpers for Cornell box construction
static void addRectangle(std::vector<Triangle> &tris, const Point3f &p0,
                         const Point3f &p1, const Point3f &p2,
                         const Point3f &p3, int material_id) {
  Triangle t1;
  t1.p0 = p0;
  t1.p1 = p1;
  t1.p2 = p2;
  t1.material_id = material_id;
  Triangle t2;
  t2.p0 = p0;
  t2.p1 = p2;
  t2.p2 = p3;
  t2.material_id = material_id;
  tris.push_back(t1);
  tris.push_back(t2);
}

// Cornell box fallback scene
void buildCornellBox(Scene &scene) {
  std::vector<Triangle> triangles;
  std::vector<Material> materials;

  // Materials
  materials.push_back(
      Material(Color3f(0.886f, 0.699f, 0.666f), Color3f(0.f))); // 0 white
  materials.push_back(
      Material(Color3f(0.105f, 0.378f, 0.076f), Color3f(0.f))); // 1 green
  materials.push_back(
      Material(Color3f(0.570f, 0.043f, 0.044f), Color3f(0.f))); // 2 red
  materials.push_back(Material(Color3f(0.886f, 0.699f, 0.666f),
                               Color3f(18.4f, 14.f, 6.8f))); // 3 light

  // Floor
  addRectangle(triangles, Point3f(-1.f, -1.f, -1.f), Point3f(1.f, -1.f, -1.f),
               Point3f(1.f, -1.f, 1.f), Point3f(-1.f, -1.f, 1.f), 0);
  // Ceiling
  addRectangle(triangles, Point3f(-1.f, 1.f, 1.f), Point3f(1.f, 1.f, 1.f),
               Point3f(1.f, 1.f, -1.f), Point3f(-1.f, 1.f, -1.f), 0);
  // Back wall
  addRectangle(triangles, Point3f(-1.f, -1.f, -1.f), Point3f(-1.f, 1.f, -1.f),
               Point3f(1.f, 1.f, -1.f), Point3f(1.f, -1.f, -1.f), 0);
  // Left wall (green)
  addRectangle(triangles, Point3f(-1.f, -1.f, 1.f), Point3f(-1.f, 1.f, 1.f),
               Point3f(-1.f, 1.f, -1.f), Point3f(-1.f, -1.f, -1.f), 1);
  // Right wall (red)
  addRectangle(triangles, Point3f(1.f, -1.f, -1.f), Point3f(1.f, 1.f, -1.f),
               Point3f(1.f, 1.f, 1.f), Point3f(1.f, -1.f, 1.f), 2);
  // Area light
  const float ls = 0.23f, lh = 0.99f;
  addRectangle(triangles, Point3f(-ls, lh, -0.19f), Point3f(ls, lh, -0.19f),
               Point3f(ls, lh, 0.19f), Point3f(-ls, lh, 0.19f), 3);

  scene.setTriangles(triangles.data(), (uint32_t)triangles.size());
  scene.setMaterials(materials.data(), (uint32_t)materials.size());
}

namespace futaba {
extern OptixDeviceContext getOptixContext();
extern void initOptix();
} // namespace futaba

struct EmptyRecord {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

class OptixPipelineManager {
public:
  OptixPipeline pipeline = nullptr;
  OptixShaderBindingTable sbt = {};
  CUdeviceptr d_params = 0;
  OptixModule module = nullptr;

  void init() {
    if (pipeline)
      return;
    futaba::initOptix();
    OptixDeviceContext context = futaba::getOptixContext();

    // 1. Pipeline Compile Options
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

    char log[2048];
    size_t sizeof_log = sizeof(log);

    optixModuleCreate(context, &moduleCompileOptions, &pipelineCompileOptions,
                      ptxCode.data(), ptxSize, log, &sizeof_log, &module);

    // 3. Program Groups
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroup raygenProgGroup;
    OptixProgramGroupDesc raygenDesc = {};
    raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDesc.raygen.module = module;
    raygenDesc.raygen.entryFunctionName = "__raygen__render";
    sizeof_log = sizeof(log);
    optixProgramGroupCreate(context, &raygenDesc, 1, &pgOptions, log,
                            &sizeof_log, &raygenProgGroup);

    OptixProgramGroup missProgGroup;
    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = module;
    missDesc.miss.entryFunctionName = "__miss__ms";
    sizeof_log = sizeof(log);
    optixProgramGroupCreate(context, &missDesc, 1, &pgOptions, log, &sizeof_log,
                            &missProgGroup);

    OptixProgramGroup hitProgGroup;
    OptixProgramGroupDesc hitDesc = {};
    hitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitDesc.hitgroup.moduleCH = module;
    hitDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    sizeof_log = sizeof(log);
    optixProgramGroupCreate(context, &hitDesc, 1, &pgOptions, log, &sizeof_log,
                            &hitProgGroup);

    // 4. Create Pipeline
    OptixProgramGroup programGroups[] = {raygenProgGroup, missProgGroup,
                                         hitProgGroup};
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 1;

    sizeof_log = sizeof(log);
    optixPipelineCreate(context, &pipelineCompileOptions, &pipelineLinkOptions,
                        programGroups, 3, log, &sizeof_log, &pipeline);

    // 5. Build SBT
    std::vector<EmptyRecord> raygenRecords(1);
    optixSbtRecordPackHeader(raygenProgGroup, &raygenRecords[0]);
    CUdeviceptr d_raygenRecord;
    cudaMalloc(reinterpret_cast<void **>(&d_raygenRecord), sizeof(EmptyRecord));
    cudaMemcpy(reinterpret_cast<void *>(d_raygenRecord), raygenRecords.data(),
               sizeof(EmptyRecord), cudaMemcpyHostToDevice);
    sbt.raygenRecord = d_raygenRecord;

    std::vector<EmptyRecord> missRecords(1);
    optixSbtRecordPackHeader(missProgGroup, &missRecords[0]);
    CUdeviceptr d_missRecord;
    cudaMalloc(reinterpret_cast<void **>(&d_missRecord), sizeof(EmptyRecord));
    cudaMemcpy(reinterpret_cast<void *>(d_missRecord), missRecords.data(),
               sizeof(EmptyRecord), cudaMemcpyHostToDevice);
    sbt.missRecordBase = d_missRecord;
    sbt.missRecordStrideInBytes = sizeof(EmptyRecord);
    sbt.missRecordCount = 1;

    std::vector<EmptyRecord> hitRecords(1);
    optixSbtRecordPackHeader(hitProgGroup, &hitRecords[0]);
    CUdeviceptr d_hitRecord;
    cudaMalloc(reinterpret_cast<void **>(&d_hitRecord), sizeof(EmptyRecord));
    cudaMemcpy(reinterpret_cast<void *>(d_hitRecord), hitRecords.data(),
               sizeof(EmptyRecord), cudaMemcpyHostToDevice);
    sbt.hitgroupRecordBase = d_hitRecord;
    sbt.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);
    sbt.hitgroupRecordCount = 1;

    cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(LaunchParams));
  }
};

static OptixPipelineManager g_pipeline;

void launch_render(uchar4 *d_buffer, HDRFilm *film, int width, int height,
                   const PerspectiveCamera &camera, const Scene &scene,
                   int max_depth, int rr_depth, int integrator_mode,
                   bool use_antialiasing) {
  g_pipeline.init();

  film->sampleCount++;

  LaunchParams params = {};
  params.pbo_ptr = d_buffer;
  params.film_pixels = film->d_pixels;
  params.width = width;
  params.height = height;
  params.sampleCount = film->sampleCount;
  params.camera = camera;
  params.scene = scene;
  params.max_depth = max_depth;
  params.rr_depth = rr_depth;
  params.integrator_mode = integrator_mode;
  params.use_antialiasing = use_antialiasing;

  cudaMemcpy(reinterpret_cast<void *>(g_pipeline.d_params), &params,
             sizeof(LaunchParams), cudaMemcpyHostToDevice);

  optixLaunch(g_pipeline.pipeline,
              0, // stream
              g_pipeline.d_params, sizeof(LaunchParams), &g_pipeline.sbt, width,
              height, 1);

  cudaDeviceSynchronize();
}
