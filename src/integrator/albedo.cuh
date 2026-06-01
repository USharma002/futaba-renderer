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

    Color3f L = si.albedo;
    return L;
  }
};

} // namespace futaba

#if !defined(__CUDACC__) && defined(NANOGUI_GLAD)
#include "integrator_ui.h"

namespace futaba {

class AlbedoIntegratorUI : public IntegratorUI {
public:
    std::string getName() const override { return "Albedo"; }
    int getMode() const override { return INTEGRATOR_ALBEDO; }
};

} // namespace futaba
#endif

