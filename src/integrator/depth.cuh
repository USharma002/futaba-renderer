#pragma once

#include "sampler.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"

FUTABA_NAMESPACE_BEGIN

// Depth integrator: returns the first surface hit distance per pixel.

struct Depth {

    HD Color3f sample(const Ray& ray, const Scene& scene,
                      Sampler& /*sampler*/) const
    {
        SurfaceIntersection si;
        scene.intersect(ray, ray.mint, ray.maxt, si);

        if (!si.is_valid())
            return Color3f(0.f);

        return Color3f(1.0/ (si.t + 1e-5f)); // Invert and add epsilon to avoid division by zero

    }
};

FUTABA_NAMESPACE_END

#if !defined(__CUDACC__) && defined(NANOGUI_GLAD)
#include "integrator_ui.h"

FUTABA_NAMESPACE_BEGIN

class DepthIntegratorUI : public IntegratorUI {
public:
    std::string getName() const override { return "Depth"; }
    int getMode() const override { return INTEGRATOR_DEPTH; }
};

FUTABA_NAMESPACE_END
#endif

