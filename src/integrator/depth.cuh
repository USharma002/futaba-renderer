#pragma once

#include "sampler.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"

namespace futaba {

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

} // namespace futaba

#if !defined(__CUDACC__) && defined(NANOGUI_GLAD)
#include "integrator_ui.h"

namespace futaba {

class DepthIntegratorUI : public IntegratorUI {
public:
    std::string getName() const override { return "Depth"; }
    int getMode() const override { return INTEGRATOR_DEPTH; }
};

} // namespace futaba
#endif

