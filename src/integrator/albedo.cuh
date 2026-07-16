#pragma once

#include "sampler.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"
#include "bsdf.cuh"


FUTABA_NAMESPACE_BEGIN

struct Albedo {
  // Visualizes surface albedo as RGB for debugging
  HD Color3f sample(const Ray &ray, const Scene &scene, Sampler &) const {
    SurfaceIntersection si;
    if (!scene.intersect(ray, ray.mint, ray.maxt, si)) {
      return Color3f(0.0f);
    }
    return BSDF::get_albedo(scene.materials[si.material_id], si);
  }
};

FUTABA_NAMESPACE_END

#if !defined(__CUDACC__) && defined(NANOGUI_GLAD)
#include "integrator_ui.h"

FUTABA_NAMESPACE_BEGIN

class AlbedoIntegratorUI : public IntegratorUI {
public:
    std::string getName() const override { return "Albedo"; }
    int getMode() const override { return INTEGRATOR_ALBEDO; }
};

FUTABA_NAMESPACE_END
#endif
