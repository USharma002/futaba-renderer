#include "common.cuh"
#include "bsdf.cuh"
#include "heatmap.cuh"
#include "albedo.cuh"
#include "phong.cuh"
#include "depth.cuh"
#include "primitives.cuh"
#include "launch_params.h"
#include "normals.cuh"
#include "path.cuh"

#include "tonemapping.cuh"
#include "perspective.cuh"
#include "scene.cuh"
#include "types.cuh"
#include <optix.h>

extern "C" {
__constant__ unsigned char params_buffer[sizeof(futaba::LaunchParams)];
}

using namespace futaba;

#define params (*reinterpret_cast<const LaunchParams *>(params_buffer))

static __device__ Ray3f setup_ray(uint3 idx, int& index, Sampler& sampler) {
  index = idx.y * params.width + idx.x;

  unsigned int seed = (unsigned int)(index + 1) ^ (params.sampleCount << 16);
  sampler = Sampler(seed);

  float jx = params.use_antialiasing ? sampler.next1D() : 0.5f;
  float jy = params.use_antialiasing ? sampler.next1D() : 0.5f;
  float u = (float)(idx.x + jx) / (float)params.width;
  float v = (float)(idx.y + jy) / (float)params.height;

  return params.camera.sampleRay(u, v, sampler);
}

// Accumulate radiance into the film buffer and, when denoising is inactive,
// tonemap and write the result directly to the PBO.
//
// When denoising is active the host-side denoiser->exec() call is responsible
// for reading film_pixels plus the albedo/normal guide buffers and writing the
// final tonemapped result to the PBO. Guide buffers are populated by the path
// kernel's PathRecorder before this function is called, so no intersection
// work is done here.
static __device__ void accumulate_and_write(int index, const Color3f& radiance) {
  Color3f acc = params.film_pixels[index];
  acc += radiance;
  params.film_pixels[index] = acc;

  if (!params.denoise.active) {
    Color3f linear_avg  = acc / (float)params.sampleCount;
    Color3f tonemapped  = tonemap::apply(linear_avg, params.tonemapping_mode);
    Color3f final_color = toSRGB(tonemapped);

    params.pbo_ptr[index].x = (unsigned char)clamp(final_color.x * 255.f, 0.f, 255.f);
    params.pbo_ptr[index].y = (unsigned char)clamp(final_color.y * 255.f, 0.f, 255.f);
    params.pbo_ptr[index].z = (unsigned char)clamp(final_color.z * 255.f, 0.f, 255.f);
    params.pbo_ptr[index].w = 255;
  }
}

extern "C" __global__ void __raygen__render() {
  uint3 idx = optixGetLaunchIndex();

  if (idx.x >= params.width || idx.y >= params.height)
    return;

  int index;
  Sampler sampler;
  Ray3f ray = setup_ray(idx, index, sampler);
  Color3f radiance;
  switch (params.integrator_mode) {
    case INTEGRATOR_NORMALS: {
      Normals normals;
      radiance = normals.sample(ray, params.scene, sampler);
      break;
    }
    case INTEGRATOR_HEATMAP: {
      Heatmap heatmap;
      radiance = heatmap.sample(ray, params.scene, sampler);
      break;
    }
    case INTEGRATOR_ALBEDO: {
      Albedo albedo;
      radiance = albedo.sample(ray, params.scene, sampler);
      break;
    }
    case INTEGRATOR_DEPTH: {
      Depth depth;
      radiance = depth.sample(ray, params.scene, sampler);
      break;
    }
    case INTEGRATOR_PHONG: {
      Phong phong(params.phong.light_dir, params.phong.ambient,
                  params.phong.diffuse, params.phong.specular,
                  params.phong.shininess);
      radiance = phong.sample(ray, params.scene, sampler);
      break;
    }
    case INTEGRATOR_PRIMITIVES:
    default: {
      Primitives primitives;
      radiance = primitives.sample(ray, params.scene, sampler);
      break;
    }
  }

  accumulate_and_write(index, radiance);
}

extern "C" __global__ void __raygen__path() {
  uint3 idx = optixGetLaunchIndex();

  if (idx.x >= params.width || idx.y >= params.height)
    return;

  int index;
  Sampler sampler;
  Ray3f ray = setup_ray(idx, index, sampler);

  EmitterSampler light_sampler(params.light_sampler);
  Path integrator(params.max_depth, params.rr_depth, light_sampler);

  PathRecorder recorder;
  bool recording = params.denoise.active && params.denoise.albedo_buffer && params.denoise.normal_buffer;
  if (recording) {
    recorder.albedo_buffer = params.denoise.albedo_buffer;
    recorder.normal_buffer = params.denoise.normal_buffer;
    recorder.pixel_index   = index;
    recorder.sample_count  = params.sampleCount;
    recorder.camera        = &params.camera;
  }

  Color3f radiance = integrator.sample(ray, params.scene, sampler, recording ? &recorder : nullptr);

  accumulate_and_write(index, radiance);
}

extern "C" __global__ void __closesthit__ch() {
  unsigned int p0 = optixGetPayload_0();
  unsigned int p1 = optixGetPayload_1();

  unsigned long long packed = (static_cast<unsigned long long>(p1) << 32) |
                              static_cast<unsigned long long>(p0);
  SurfaceIntersection *rec = reinterpret_cast<SurfaceIntersection *>(packed);

  unsigned int primIdx = optixGetPrimitiveIndex();
  const Triangle &tri = params.scene.triangles[primIdx];

  Ray3f ray(Point3f(optixGetWorldRayOrigin().x, optixGetWorldRayOrigin().y,
                    optixGetWorldRayOrigin().z),
            Vector3f(optixGetWorldRayDirection().x,
                     optixGetWorldRayDirection().y,
                     optixGetWorldRayDirection().z));

  float2 bary = optixGetTriangleBarycentrics();
  tri.populate_intersection(ray, optixGetRayTmax(), bary.x, bary.y, *rec,
                            params.scene.use_vertex_normals, (int)primIdx);
}

extern "C" __global__ void __miss__ms() {
  // bvh.intersect handles misses by checking rec.is_valid().
  // We don't need to do anything here.
}

extern "C" __global__ void __anyhit__shadow() {
  unsigned int target_mesh_id = optixGetPayload_0();
  unsigned int primIdx        = optixGetPrimitiveIndex();
  const Triangle &tri         = params.scene.triangles[primIdx];

  if (tri.mesh_id == (int)target_mesh_id) {
    optixIgnoreIntersection();
  }

  if (tri.material_id >= 0 && tri.material_id < (int)params.scene.materialCount) {
    int mat_type = params.scene.materials[tri.material_id].type;
    if (futaba::Material::isShadowTransparent((futaba::BSDFType)mat_type)) {
      optixIgnoreIntersection();
    }
  }

  optixSetPayload_1(1);
  optixTerminateRay();
}

extern "C" __global__ void __miss__shadow() {
  optixSetPayload_1(0);
}
