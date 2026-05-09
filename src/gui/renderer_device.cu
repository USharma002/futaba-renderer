#include "common.cuh"
#include "heatmap.cuh"
#include "launch_params.h"
#include "normals.cuh"
#include "path.cuh"
#include "perspective.cuh"
#include "scene.cuh"
#include "types.cuh"
#include <optix.h>

extern "C" {
__constant__ unsigned char params_buffer[sizeof(futaba::LaunchParams)];
}

using namespace futaba;

#define params (*reinterpret_cast<const LaunchParams *>(params_buffer))

extern "C" __global__ void __raygen__render() {
  uint3 idx = optixGetLaunchIndex();

  if (idx.x >= params.width || idx.y >= params.height)
    return;

  int index = idx.y * params.width + idx.x;

  unsigned int seed = (unsigned int)(index + 1) ^ (params.sampleCount << 16);
  Sampler sampler(seed);

  // If anti aliasing is enabled, use random jitter esle 0.5f to sample pixel
  // center
  float jx = params.use_antialiasing ? sampler.next1D() : 0.5f;
  float jy = params.use_antialiasing ? sampler.next1D() : 0.5f;
  float u = (float)(idx.x + jx) / (float)params.width;
  float v = (float)(idx.y + jy) / (float)params.height;

  Ray3f ray = params.camera.sampleRay(u, v);

  // Select correct integrator based on mode in GUI
  Color3f radiance;
  if (params.integrator_mode == INTEGRATOR_NORMALS) {
    Normals normals;
    radiance = normals.sample(ray, params.scene, sampler);
  } else if (params.integrator_mode == INTEGRATOR_HEATMAP) {
    Heatmap heatmap(16);
    radiance = heatmap.sample(ray, params.scene, sampler);
  } else {
    Path integrator(params.max_depth, params.rr_depth);
    radiance = integrator.sample(ray, params.scene, sampler);
  }

  Color3f acc = params.film_pixels[index];
  acc += radiance;
  params.film_pixels[index] = acc;

  Color3f final_color = toSRGB(acc / (float)params.sampleCount);

  params.pbo_ptr[index].x =
      (unsigned char)clamp(final_color.x * 255.f, 0.f, 255.f);
  params.pbo_ptr[index].y =
      (unsigned char)clamp(final_color.y * 255.f, 0.f, 255.f);
  params.pbo_ptr[index].z =
      (unsigned char)clamp(final_color.z * 255.f, 0.f, 255.f);
  params.pbo_ptr[index].w = 255;
}

extern "C" __global__ void __closesthit__ch() {
  unsigned int p0 = optixGetPayload_0();
  unsigned int p1 = optixGetPayload_1();

  unsigned long long packed = (static_cast<unsigned long long>(p1) << 32) |
                              static_cast<unsigned long long>(p0);
  SurfaceIntersection *rec = reinterpret_cast<SurfaceIntersection *>(packed);

  unsigned int primIdx = optixGetPrimitiveIndex();
  const Triangle &tri = params.scene.triangles[primIdx];

  // Evaluate full triangle intersection properties
  Ray3f ray(Point3f(optixGetWorldRayOrigin().x, optixGetWorldRayOrigin().y,
                    optixGetWorldRayOrigin().z),
            Vector3f(optixGetWorldRayDirection().x,
                     optixGetWorldRayDirection().y,
                     optixGetWorldRayDirection().z));

  // Since OptiX reported a hit, we know it intersects. We pass bounds that
  // ensure Triangle::intersect completes successfully.
  tri.intersect(ray, 0.0f, optixGetRayTmax() + 0.001f, *rec,
                params.scene.use_vertex_normals);
}

extern "C" __global__ void __miss__ms() {
  // bvh.intersect handles misses by checking rec.is_valid().
  // We don't need to do anything here.
}
