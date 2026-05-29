#include "launch_params.h"
#include "renderer.h"
#include "distribution.cuh"
#include "denoiser.h"
#include "training_buffer.h"
#include <iostream>
#include <optix.h>
#include <optix_stubs.h>
#include <vector>
#include <atomic>

namespace futaba {
    std::atomic<float> g_optixCompileProgress(0.0f);
    std::atomic<const char*> g_optixCompileStatus("Starting OptiX compilation...");
    std::atomic<bool> g_optixCompileCompleted(false);
}

using namespace futaba;

// Geometry helpers for Cornell box construction
static void addRectangle(std::vector<Triangle> &tris, const Point3f &p0,
                         const Point3f &p1, const Point3f &p2,
                         const Point3f &p3, int material_id, int mesh_id) {
  Vector3f n = normalize(cross(p1 - p0, p2 - p0));
  Triangle t1;
  t1.p0 = p0; t1.p1 = p1; t1.p2 = p2;
  t1.n0 = n; t1.n1 = n; t1.n2 = n;
  t1.has_normals = true;
  t1.material_id = material_id;
  t1.mesh_id = mesh_id;
  Triangle t2;
  t2.p0 = p0; t2.p1 = p2; t2.p2 = p3;
  t2.n0 = n; t2.n1 = n; t2.n2 = n;
  t2.has_normals = true;
  t2.material_id = material_id;
  t2.mesh_id = mesh_id;
  tris.push_back(t1);
  tris.push_back(t2);
}

// Cornell box fallback scene — now builds proper emitter data for NEE.
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

  // Each surface gets its own mesh_id so it maps to its MeshInstanceGPU.
  //   mesh 0 = floor, mesh 1 = ceiling, mesh 2 = back wall,
  //   mesh 3 = left wall, mesh 4 = right wall, mesh 5 = light
  addRectangle(triangles, Point3f(-1.f, -1.f, -1.f), Point3f(1.f, -1.f, -1.f),
               Point3f(1.f, -1.f, 1.f), Point3f(-1.f, -1.f, 1.f), 0, 0);
  addRectangle(triangles, Point3f(-1.f, 1.f, 1.f), Point3f(1.f, 1.f, 1.f),
               Point3f(1.f, 1.f, -1.f), Point3f(-1.f, 1.f, -1.f), 0, 1);
  addRectangle(triangles, Point3f(-1.f, -1.f, -1.f), Point3f(-1.f, 1.f, -1.f),
               Point3f(1.f, 1.f, -1.f), Point3f(1.f, -1.f, -1.f), 0, 2);
  addRectangle(triangles, Point3f(-1.f, -1.f, 1.f), Point3f(-1.f, 1.f, 1.f),
               Point3f(-1.f, 1.f, -1.f), Point3f(-1.f, -1.f, -1.f), 1, 3);
  addRectangle(triangles, Point3f(1.f, -1.f, -1.f), Point3f(1.f, 1.f, -1.f),
               Point3f(1.f, 1.f, 1.f), Point3f(1.f, -1.f, 1.f), 2, 4);
  // Area light
  const float ls = 0.23f, lh = 0.99f;
  addRectangle(triangles, Point3f(-ls, lh, -0.19f), Point3f(ls, lh, -0.19f),
               Point3f(ls, lh, 0.19f), Point3f(-ls, lh, 0.19f), 3, 5);

  scene.setTriangles(triangles.data(), (uint32_t)triangles.size());
  scene.setMaterials(materials.data(), (uint32_t)materials.size());

  // ---------------------------------------------------------------------------
  // Mesh instances — one per logical surface, light mesh gets an emitter
  // ---------------------------------------------------------------------------
  const int EMITTER_ID_LIGHT = 0; // index into emitters array below

  std::vector<futaba::MeshInstanceGPU> meshes;
  // meshes 0..4: non-emissive surfaces (2 triangles each)
  for (int i = 0; i < 5; ++i) {
    futaba::MeshInstanceGPU m;
    m.triangleStart = (uint32_t)(i * 2);
    m.triangleCount = 2;
    m.emitterId = -1;
    meshes.push_back(m);
  }
  // mesh 5: light panel
  {
    futaba::MeshInstanceGPU m;
    m.triangleStart = 10; // triangles 10, 11
    m.triangleCount = 2;
    m.emitterId = EMITTER_ID_LIGHT;
    meshes.push_back(m);
  }
  scene.setMeshes(meshes.data(), (uint32_t)meshes.size());

  // ---------------------------------------------------------------------------
  // Emitter records
  // ---------------------------------------------------------------------------
  futaba::EmitterGPU lightEm;
  lightEm.type           = futaba::kEmitterTypeArea;
  lightEm.flags          = futaba::EMITTER_FLAG_TWO_SIDED;
  lightEm.radiance       = Color3f(18.4f, 14.f, 6.8f);
  lightEm.position       = Point3f(0.f);
  lightEm.direction      = Vector3f(0.f);
  lightEm.attachedMeshId = 5;
  scene.setEmitters(&lightEm, 1);

  // ---------------------------------------------------------------------------
  // Build emissive-triangle CDF (same algorithm as gui.cpp::loadScene)
  // ---------------------------------------------------------------------------
  auto luminance = [](const Color3f &c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
  };

  std::vector<int>   emissiveTriIndices;
  std::vector<float> emissiveWeights;

  for (size_t i = 0; i < triangles.size(); ++i) {
    const Triangle &t = triangles[i];
    Color3f emission(0.f);
    // Check if triangle belongs to a mesh with an emitter
    if (t.mesh_id >= 0 && t.mesh_id < (int)meshes.size()) {
      int eid = meshes[t.mesh_id].emitterId;
      if (eid == EMITTER_ID_LIGHT)
        emission = lightEm.radiance;
    }
    // Fallback: check material emission
    if (emission.x <= 0.f && emission.y <= 0.f && emission.z <= 0.f) {
      if (t.material_id >= 0 && t.material_id < (int)materials.size())
        emission = materials[t.material_id].emission;
    }
    float w = t.area() * luminance(emission);
    if (w > 0.f) {
      emissiveTriIndices.push_back((int)i);
      emissiveWeights.push_back(w);
    }
  }

  if (!emissiveWeights.empty()) {
    futaba::Distribution1D dist;
    dist.build(emissiveWeights);
    std::vector<int> globalToEmissive(triangles.size(), -1);
    for (size_t i = 0; i < emissiveTriIndices.size(); ++i) {
      int g = emissiveTriIndices[i];
      if (g >= 0 && g < (int)globalToEmissive.size())
        globalToEmissive[g] = (int)i;
    }
    scene.setEmitterTriangleDistribution(
        dist.cdfData(), (int)dist.cdf.size(), dist.funcSum,
        emissiveTriIndices.data(), (int)emissiveTriIndices.size(),
        globalToEmissive.data(), (int)globalToEmissive.size());
  } else {
    scene.setEmitterTriangleDistribution(nullptr, 0, 0.f, nullptr, 0,
                                         nullptr, 0);
  }

  // No non-area emitters in the Cornell box
  scene.setNonAreaEmitters(nullptr, 0);
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
  CUdeviceptr d_raygenRecordsBase = 0;

  void init() {
    if (pipeline)
      return;
    
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

    char log[2048];
    size_t sizeof_log = sizeof(log);

    futaba::g_optixCompileProgress = 0.20f;
    futaba::g_optixCompileStatus = "Compiling OptiX device module...";
    optixModuleCreate(context, &moduleCompileOptions, &pipelineCompileOptions,
                      ptxCode.data(), ptxSize, log, &sizeof_log, &module);

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
    optixProgramGroupCreate(context, &raygenDescRender, 1, &pgOptions, log,
                            &sizeof_log, &raygenProgGroupRender);

    futaba::g_optixCompileProgress = 0.40f;
    futaba::g_optixCompileStatus = "Creating shader programs (2/7)...";
    OptixProgramGroup raygenProgGroupPath;
    OptixProgramGroupDesc raygenDescPath = {};
    raygenDescPath.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDescPath.raygen.module = module;
    raygenDescPath.raygen.entryFunctionName = "__raygen__path";
    sizeof_log = sizeof(log);
    optixProgramGroupCreate(context, &raygenDescPath, 1, &pgOptions, log,
                            &sizeof_log, &raygenProgGroupPath);

    futaba::g_optixCompileProgress = 0.45f;
    futaba::g_optixCompileStatus = "Creating shader programs (3/7)...";
    OptixProgramGroup raygenProgGroupVolPath;
    OptixProgramGroupDesc raygenDescVolPath = {};
    raygenDescVolPath.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDescVolPath.raygen.module = module;
    raygenDescVolPath.raygen.entryFunctionName = "__raygen__volpath";
    sizeof_log = sizeof(log);
    optixProgramGroupCreate(context, &raygenDescVolPath, 1, &pgOptions, log,
                            &sizeof_log, &raygenProgGroupVolPath);

    futaba::g_optixCompileProgress = 0.50f;
    futaba::g_optixCompileStatus = "Creating shader programs (4/7)...";
    OptixProgramGroup missProgGroup;
    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = module;
    missDesc.miss.entryFunctionName = "__miss__ms";
    sizeof_log = sizeof(log);
    optixProgramGroupCreate(context, &missDesc, 1, &pgOptions, log, &sizeof_log,
                            &missProgGroup);

    futaba::g_optixCompileProgress = 0.55f;
    futaba::g_optixCompileStatus = "Creating shader programs (5/7)...";
    OptixProgramGroup hitProgGroup;
    OptixProgramGroupDesc hitDesc = {};
    hitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitDesc.hitgroup.moduleCH = module;
    hitDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    sizeof_log = sizeof(log);
    optixProgramGroupCreate(context, &hitDesc, 1, &pgOptions, log, &sizeof_log,
                            &hitProgGroup);

    // Shadow miss
    futaba::g_optixCompileProgress = 0.60f;
    futaba::g_optixCompileStatus = "Creating shader programs (6/7)...";
    OptixProgramGroup shadowMissProgGroup;
    OptixProgramGroupDesc shadowMissDesc = {};
    shadowMissDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    shadowMissDesc.miss.module = module;
    shadowMissDesc.miss.entryFunctionName = "__miss__shadow";
    sizeof_log = sizeof(log);
    optixProgramGroupCreate(context, &shadowMissDesc, 1, &pgOptions, log,
                            &sizeof_log, &shadowMissProgGroup);

    // Shadow hit (anyhit only, no closest-hit)
    futaba::g_optixCompileProgress = 0.65f;
    futaba::g_optixCompileStatus = "Creating shader programs (7/7)...";
    OptixProgramGroup shadowHitProgGroup;
    OptixProgramGroupDesc shadowHitDesc = {};
    shadowHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    shadowHitDesc.hitgroup.moduleAH = module;
    shadowHitDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";
    sizeof_log = sizeof(log);
    optixProgramGroupCreate(context, &shadowHitDesc, 1, &pgOptions, log,
                            &sizeof_log, &shadowHitProgGroup);

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
    optixPipelineCreate(context, &pipelineCompileOptions, &pipelineLinkOptions,
                        programGroups, 7, log, &sizeof_log, &pipeline);

    // 5. Build SBT
    futaba::g_optixCompileProgress = 0.90f;
    futaba::g_optixCompileStatus = "Building Shader Binding Table (SBT)...";
    std::vector<EmptyRecord> raygenRecords(3);
    optixSbtRecordPackHeader(raygenProgGroupRender, &raygenRecords[0]);
    optixSbtRecordPackHeader(raygenProgGroupPath, &raygenRecords[1]);
    optixSbtRecordPackHeader(raygenProgGroupVolPath, &raygenRecords[2]);
    cudaMalloc(reinterpret_cast<void **>(&d_raygenRecordsBase), 3 * sizeof(EmptyRecord));
    cudaMemcpy(reinterpret_cast<void *>(d_raygenRecordsBase), raygenRecords.data(),
               3 * sizeof(EmptyRecord), cudaMemcpyHostToDevice);
    sbt.raygenRecord = d_raygenRecordsBase;

    std::vector<EmptyRecord> missRecords(2);
    optixSbtRecordPackHeader(missProgGroup, &missRecords[0]);
    optixSbtRecordPackHeader(shadowMissProgGroup, &missRecords[1]);
    CUdeviceptr d_missRecord;
    cudaMalloc(reinterpret_cast<void **>(&d_missRecord),
               2 * sizeof(EmptyRecord));
    cudaMemcpy(reinterpret_cast<void *>(d_missRecord), missRecords.data(),
               2 * sizeof(EmptyRecord), cudaMemcpyHostToDevice);
    sbt.missRecordBase = d_missRecord;
    sbt.missRecordStrideInBytes = sizeof(EmptyRecord);
    sbt.missRecordCount = 2;

    std::vector<EmptyRecord> hitRecords(2);
    optixSbtRecordPackHeader(hitProgGroup, &hitRecords[0]);
    optixSbtRecordPackHeader(shadowHitProgGroup, &hitRecords[1]);
    CUdeviceptr d_hitRecord;
    cudaMalloc(reinterpret_cast<void **>(&d_hitRecord),
               2 * sizeof(EmptyRecord));
    cudaMemcpy(reinterpret_cast<void *>(d_hitRecord), hitRecords.data(),
               2 * sizeof(EmptyRecord), cudaMemcpyHostToDevice);
    sbt.hitgroupRecordBase = d_hitRecord;
    sbt.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);
    sbt.hitgroupRecordCount = 2;

    cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(LaunchParams));

    futaba::g_optixCompileProgress = 1.00f;
    futaba::g_optixCompileStatus = "Completed";
    futaba::g_optixCompileCompleted = true;
  }
};

static OptixPipelineManager g_pipeline;

namespace futaba {
    void launch_initial_pipeline_compile() {
        g_pipeline.init();
    }
}

void launch_render(HDRFilm *film,
                   DenoiserManager* denoiser,
                   LaunchParams params) {
  g_pipeline.init();

  film->sampleCount++;

  params.film_pixels = film->d_pixels;
  params.sampleCount = film->sampleCount;
  
  if (params.denoise_active && denoiser) {
    params.denoise_albedo_buffer = denoiser->getAlbedoBuffer();
    params.denoise_normal_buffer = denoiser->getNormalBuffer();
  } else {
    params.denoise_albedo_buffer = nullptr;
    params.denoise_normal_buffer = nullptr;
  }

  cudaMemcpy(reinterpret_cast<void *>(g_pipeline.d_params), &params,
             sizeof(LaunchParams), cudaMemcpyHostToDevice);

  int raygen_idx = 0; // default to render/preview
  if (params.integrator_mode == INTEGRATOR_PATH) {
      raygen_idx = 1;
  } else if (params.integrator_mode == INTEGRATOR_VOLPATH) {
      raygen_idx = 2;
  }
  g_pipeline.sbt.raygenRecord = g_pipeline.d_raygenRecordsBase + raygen_idx * sizeof(EmptyRecord);

  optixLaunch(g_pipeline.pipeline,
              0, // stream
              g_pipeline.d_params, sizeof(LaunchParams), &g_pipeline.sbt, params.width,
              params.height, 1);

  // If denoising is active, execute the denoiser pipeline (which computes autoexposure, denoises, tonemaps, and copies to PBO)
  if (params.denoise_active && denoiser) {
    denoiser->exec(
        film->d_pixels,
        denoiser->getAlbedoBuffer(),
        denoiser->getNormalBuffer(),
        film->sampleCount,
        params.pbo_ptr,
        params.tonemapping_mode
    );
  }

  // Run training visualization kernel if active
  if (params.vis_active && params.vis_pbo_ptr) {
    run_visualization_kernel(
        params.train_active,
        params.train_position,
        params.train_normals,
        params.train_wi,
        params.train_wo,
        params.train_radiance,
        params.train_material_id,
        params.width, params.height,
        params.max_depth,
        params.vis_depth,
        params.vis_buffer_type,
        params.vis_pbo_ptr
    );
  }

  cudaDeviceSynchronize();
}
