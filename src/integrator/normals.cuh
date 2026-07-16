#pragma once

#include "sampler.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"

FUTABA_NAMESPACE_BEGIN

// Normals integrator: visualises shading normals as RGB for debugging.
// The sampler parameter is accepted for interface uniformity but is not used
// (normal visualisation is deterministic).
struct Normals {
    HD Color3f sample(const Ray& ray, const Scene& scene,
                      Sampler& /*sampler*/) const
    {
        SurfaceIntersection si;
        if (!scene.intersect(ray, ray.mint, ray.maxt, si))
            return Color3f(0.f);

        // Remap normal components from [-1, 1] to [0, 1].
        return 0.5f * (normalize(si.n) + Vector3f(1.f));
    }
};

FUTABA_NAMESPACE_END

#if !defined(__CUDACC__) && defined(NANOGUI_GLAD)
#include "integrator_ui.h"

FUTABA_NAMESPACE_BEGIN

class NormalsIntegratorUI : public IntegratorUI {
public:
    std::string getName() const override { return "Normals"; }
    int getMode() const override { return INTEGRATOR_NORMALS; }
};

FUTABA_NAMESPACE_END
#endif

