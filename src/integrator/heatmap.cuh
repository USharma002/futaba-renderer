#pragma once

#include "sampler.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"

namespace futaba {

struct Heatmap {
  // Count ray-geometry intersections via multisampling along view direction.
  // Shoots N rays with angular jitter from primary ray origin to estimate
  // local geometry complexity and intersection density.

  int samples_per_pixel;

  HD Heatmap(int spp = 8) : samples_per_pixel(spp) {}

  HD Color3f sample(const Ray &ray, const Scene &scene,
                    Sampler &sampler) const {
    int aabb_tests = scene.intersectAABBCount(ray, ray.mint, ray.maxt);

    // Without culling, the number of AABB intersections is much higher.
    // Let's cap at around 128 for normalization.
    float normalized = fminf((float)aabb_tests / 512.f, 1.f);

    return custom_colormap(normalized);
  }

private:
  HD Color3f custom_colormap(float t) const {
    t = fmaxf(0.f, fminf(1.f, t));

    // Colormap matching the provided image: Dark Blue -> Magenta -> Pink ->
    // White
    if (t < 0.25f) {
      float u = t / 0.25f;
      return Color3f(0.f, 0.f, u * 0.6f); // Black to Dark Blue
    } else if (t < 0.5f) {
      float u = (t - 0.25f) / 0.25f;
      return Color3f(u * 0.8f, 0.f, 0.6f + u * 0.4f); // Dark Blue to Magenta
    } else if (t < 0.75f) {
      float u = (t - 0.5f) / 0.25f;
      return Color3f(0.8f + u * 0.2f, u * 0.3f,
                     1.0f - u * 0.2f); // Magenta to Pink
    } else {
      float u = (t - 0.75f) / 0.25f;
      return Color3f(1.0f, 0.3f + u * 0.7f, 0.8f + u * 0.2f); // Pink to White
    }
  }
};

} // namespace futaba
