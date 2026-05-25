#include "launch_params.h"
#include "renderer.h"
#include "distribution.cuh"
#include "denoiser.h"
#include "training_buffer.h"
#include <iostream>
#include <optix.h>
#include <optix_stubs.h>
#include <vector>

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

    // Shadow miss
    OptixProgramGroup shadowMissProgGroup;
    OptixProgramGroupDesc shadowMissDesc = {};
    shadowMissDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    shadowMissDesc.miss.module = module;
    shadowMissDesc.miss.entryFunctionName = "__miss__shadow";
    sizeof_log = sizeof(log);
    optixProgramGroupCreate(context, &shadowMissDesc, 1, &pgOptions, log,
                            &sizeof_log, &shadowMissProgGroup);

    // Shadow hit (anyhit only, no closest-hit)
    OptixProgramGroup shadowHitProgGroup;
    OptixProgramGroupDesc shadowHitDesc = {};
    shadowHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    shadowHitDesc.hitgroup.moduleAH = module;
    shadowHitDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";
    sizeof_log = sizeof(log);
    optixProgramGroupCreate(context, &shadowHitDesc, 1, &pgOptions, log,
                            &sizeof_log, &shadowHitProgGroup);

    // 4. Create Pipeline
    OptixProgramGroup programGroups[] = {raygenProgGroup, missProgGroup,
                                         hitProgGroup, shadowMissProgGroup,
                                         shadowHitProgGroup};
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 1;

    sizeof_log = sizeof(log);
    optixPipelineCreate(context, &pipelineCompileOptions, &pipelineLinkOptions,
                        programGroups, 5, log, &sizeof_log, &pipeline);

    // 5. Build SBT
    std::vector<EmptyRecord> raygenRecords(1);
    optixSbtRecordPackHeader(raygenProgGroup, &raygenRecords[0]);
    CUdeviceptr d_raygenRecord;
    cudaMalloc(reinterpret_cast<void **>(&d_raygenRecord), sizeof(EmptyRecord));
    cudaMemcpy(reinterpret_cast<void *>(d_raygenRecord), raygenRecords.data(),
               sizeof(EmptyRecord), cudaMemcpyHostToDevice);
    sbt.raygenRecord = d_raygenRecord;

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
  }
};

static OptixPipelineManager g_pipeline;

void launch_render(uchar4 *d_buffer, HDRFilm *film, int width, int height,
                   const PerspectiveCamera &camera, const Scene &scene,
                   int max_depth, int rr_depth, int integrator_mode,
                   int tonemapping_mode, bool use_antialiasing,
                   const ::Vector3f &phong_light_dir,
                   float phong_ambient, float phong_diffuse,
                   float phong_specular, float phong_shininess,
                   bool use_denoiser,
                   futaba::DenoiserManager* denoiser,
                   int path_guiding_mode,
                   // Training buffers
                   float* train_active,
                   Point3f* train_position,
                   Color3f* train_normals,
                   Color3f* train_wi,
                   Color3f* train_wo,
                   Color3f* train_radiance,
                   float* train_material_id,
                   // Visualization parameters
                   uchar4* d_vis_buffer,
                   int vis_depth,
                   int vis_buffer_type,
                   bool vis_active) {
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
  params.tonemapping_mode = tonemapping_mode;
  params.use_antialiasing = use_antialiasing;
  params.phong_light_dir = phong_light_dir;
  params.phong_ambient = phong_ambient;
  params.phong_diffuse = phong_diffuse;
  params.phong_specular = phong_specular;
  params.phong_shininess = phong_shininess;
  params.denoise_active = use_denoiser;
  params.path_guiding_mode = path_guiding_mode;
  
  if (use_denoiser && denoiser) {
    params.denoise_albedo_buffer = denoiser->getAlbedoBuffer();
    params.denoise_normal_buffer = denoiser->getNormalBuffer();
  } else {
    params.denoise_albedo_buffer = nullptr;
    params.denoise_normal_buffer = nullptr;
  }

  // Training parameters
  params.train_active = train_active;
  params.train_position = train_position;
  params.train_normals = train_normals;
  params.train_wi = train_wi;
  params.train_wo = train_wo;
  params.train_radiance = train_radiance;
  params.train_material_id = train_material_id;

  // Visualization parameters
  params.vis_pbo_ptr = d_vis_buffer;
  params.vis_depth = vis_depth;
  params.vis_buffer_type = vis_buffer_type;

  cudaMemcpy(reinterpret_cast<void *>(g_pipeline.d_params), &params,
             sizeof(LaunchParams), cudaMemcpyHostToDevice);

  optixLaunch(g_pipeline.pipeline,
              0, // stream
              g_pipeline.d_params, sizeof(LaunchParams), &g_pipeline.sbt, width,
              height, 1);

  cudaDeviceSynchronize();

  // If denoising is active, execute the denoiser pipeline (which computes autoexposure, denoises, tonemaps, and copies to PBO)
  if (use_denoiser && denoiser) {
    denoiser->exec(
        film->d_pixels,
        denoiser->getAlbedoBuffer(),
        denoiser->getNormalBuffer(),
        film->sampleCount,
        d_buffer,
        tonemapping_mode
    );
  }

  // Run training visualization kernel if active
  if (vis_active && d_vis_buffer) {
    run_visualization_kernel(
        train_active,
        train_position,
        train_normals,
        train_wi,
        train_wo,
        train_radiance,
        train_material_id,
        width, height,
        max_depth,
        vis_depth,
        vis_buffer_type,
        d_vis_buffer
    );
  }
}
