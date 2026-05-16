#pragma once

#include "sampler.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"


namespace futaba {

struct Albedo {
  // Visualizes surface albedo as RGB for debugging
  HD Color3f sample(const Ray &ray, const Scene &scene, Sampler &) const {
    SurfaceIntersection si;
    if (!scene.intersect(ray, ray.mint, ray.maxt, si)) {
      return Color3f(0.0f);
    }

    Vector3f n = normalize(si.n);
    if (dot(ray.d, n) > 0.0f)
      n = -n;

    // TODO, not yet sure if I want to add point light sources in the scene

    Color3f L = si.albedo; // Color3f(0.5f * (n.x + 1.0f), 0.5f * (n.y + 1.0f),
                   // 0.5f * (n.z + 1.0f));
    return L; // Color3f(0.5f * (n.x + 1.0f), 0.5f * (n.y + 1.0f),
                   // 0.5f * (n.z + 1.0f));
  }
};

} // namespace futaba
