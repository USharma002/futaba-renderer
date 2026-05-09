#pragma once

#include "sampler.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"


namespace futaba {

struct Phong {
  // Visualizes surface normals as RGB for debugging
  HD Color3f sample(const Ray &ray, const Scene &scene, Sampler &) const {
    SurfaceIntersection si;
    if (!scene.intersect(ray, ray.mint, ray.maxt, si)) {
      return Color3f(0.0f);
    }

    Vector3f n = normalize(si.n);
    if (dot(current_ray.d, n) > 0.0f)
      n = -n;

    Vector3f lightDir = Vector3f(1.0, 1.0, 1.0);
    // TODO, not yet sure if I want to add point light sources in the scene

    return Color3f(0.5f * (n.x + 1.0f), 0.5f * (n.y + 1.0f),
                   0.5f * (n.z + 1.0f));
  }
};

} // namespace futaba
