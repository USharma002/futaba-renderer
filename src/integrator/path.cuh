#pragma once

#include "bsdf_sample.cuh"
#include "sampler.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"
#include <cmath>


namespace futaba {

struct Path {
  int max_depth;
  int rr_depth;

  HD Path(int max_d = 5, int rr_d = 5) : max_depth(max_d), rr_depth(rr_d) {}

  HD Color3f sample(const Ray &ray, const Scene &scene,
                    Sampler &sampler) const {
    // Set up state variables

    Color3f L(0.0f); // Radiance accumulator
    Color3f f(1.0f); // Path throughput weight

    Ray current_ray = ray; // Current Ray

    for (int depth = 0; depth < max_depth; depth++) {
      // Compute a surface interaciton with the given ray
      SurfaceIntersection si;
      bool hit =
          scene.intersect(current_ray, current_ray.mint, current_ray.maxt, si);

      // If ray missed, break the loop
      if (!hit) {
        break;
      }

      // Direct Emissiion
      L += f * si.emission;

      // Sample the BSDF for this hit record.
      BSDFSample bs;
      si.sample_bsdf(bs, sampler.next2D());
      if (bs.pdf <= 0.0f)
        break;

      Vector3f wo_world = si.to_world(bs.wo);

      f *= bs.weight;
      current_ray = si.spawn_ray(wo_world);

      // Russian roulette stopping probability
      float max_f = fmaxf(f.x, fmaxf(f.y, f.z));
      if (max_f <= 0.0f)
        break;

      if (depth >= rr_depth) {
        float rr_prob = fminf(max_f, 0.95f);
        if (sampler.next1D() >= rr_prob)
          break;
        f *= 1.0f / rr_prob;
      }
    }

    return L;
  }
};

} // namespace futaba