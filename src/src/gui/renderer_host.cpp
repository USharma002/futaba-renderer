#include "launch_params.h"
#include "renderer.h"
#include "distribution.cuh"
#include "denoiser.h"
#include "training_buffer.h"
#include "scene_uploader.h"
#include "scene_loader.h"
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
  LoadedScene loaded;

  // Materials
  loaded.materials.push_back(
      Material(Color3f(0.886f, 0.699f, 0.666f), Color3f(0.f))); // 0 white
  loaded.materials.push_back(
      Material(Color3f(0.105f, 0.378f, 0.076f), Color3f(0.f))); // 1 green
  loaded.materials.push_back(
      Material(Color3f(0.570f, 0.043f, 0.044f), Color3f(0.f))); // 2 red
  loaded.materials.push_back(Material(Color3f(0.886f, 0.699f, 0.666f),
                               Color3f(18.4f, 14.f, 6.8f))); // 3 light
  loaded.materialTexturePaths.resize(4);

  // Each surface gets its own mesh_id so it maps to its MeshInstanceGPU.
  //   mesh 0 = floor, mesh 1 = ceiling, mesh 2 = back wall,
  //   mesh 3 = left wall, mesh 4 = right wall, mesh 5 = light
  addRectangle(loaded.triangles, Point3f(-1.f, -1.f, -1.f), Point3f(1.f, -1.f, -1.f),
               Point3f(1.f, -1.f, 1.f), Point3f(-1.f, -1.f, 1.f), 0, 0);
  addRectangle(loaded.triangles, Point3f(-1.f, 1.f, 1.f), Point3f(1.f, 1.f, 1.f),
               Point3f(1.f, 1.f, -1.f), Point3f(-1.f, 1.f, -1.f), 0, 1);
  addRectangle(loaded.triangles, Point3f(-1.f, -1.f, -1.f), Point3f(-1.f, 1.f, -1.f),
               Point3f(1.f, 1.f, -1.f), Point3f(1.f, -1.f, -1.f), 0, 2);
  addRectangle(loaded.triangles, Point3f(-1.f, -1.f, 1.f), Point3f(-1.f, 1.f, 1.f),
               Point3f(-1.f, 1.f, -1.f), Point3f(-1.f, -1.f, -1.f), 1, 3);
  addRectangle(loaded.triangles, Point3f(1.f, -1.f, -1.f), Point3f(1.f, 1.f, -1.f),
               Point3f(1.f, 1.f, 1.f), Point3f(1.f, -1.f, 1.f), 2, 4);
  // Area light
  const float ls = 0.23f, lh = 0.99f;
  addRectangle(loaded.triangles, Point3f(-ls, lh, -0.19f), Point3f(ls, lh, -0.19f),
               Point3f(ls, lh, 0.19f), Point3f(-ls, lh, 0.19f), 3, 5);

  // Emitters
  EmitterInstance lightEm(EmitterType::Area);
  lightEm.radiance = Color3f(18.4f, 14.f, 6.8f);
  lightEm.twoSided = true;
  loaded.emitters.push_back(lightEm);

  // Meshes
  for (int i = 0; i < 5; ++i) {
    MeshInstance m;
    m.triangleStart = (uint32_t)(i * 2);
    m.triangleCount = 2;
    m.emitterId = -1;
    m.materialId = (i == 3) ? 1 : ((i == 4) ? 2 : 0);
    loaded.meshes.push_back(m);
  }
  {
    MeshInstance m;
    m.triangleStart = 10;
    m.triangleCount = 2;
    m.emitterId = 0;
    m.materialId = 3;
    loaded.meshes.push_back(m);
  }

  TextureManager dummyManager;
  SceneUploader::upload(loaded, "", scene, dummyManager, true, true);
}

#include "optix_pipeline.h"

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

  // Copy parameters asynchronously to the GPU using the render stream
  cudaMemcpyAsync(g_pipeline.d_params.get(), &params,
                  sizeof(LaunchParams), cudaMemcpyHostToDevice, g_pipeline.renderStream);

  int raygen_idx = 0; // default to render/preview
  if (params.integrator_mode == INTEGRATOR_PATH) {
      raygen_idx = 1;
  } else if (params.integrator_mode == INTEGRATOR_VOLPATH) {
      raygen_idx = 2;
  }
  g_pipeline.sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(
      g_pipeline.d_raygenRecordsBase.get() + raygen_idx * sizeof(EmptyRecord)
  );

  // Launch OptiX on the render stream
  optixLaunch(g_pipeline.pipeline,
              g_pipeline.renderStream,
              reinterpret_cast<CUdeviceptr>(g_pipeline.d_params.get()), sizeof(LaunchParams),
              &g_pipeline.sbt, params.width, params.height, 1);

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

  // Synchronize device to ensure rendering and memory copies are completed
  cudaDeviceSynchronize();
}