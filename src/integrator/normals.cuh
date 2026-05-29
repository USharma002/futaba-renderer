#pragma once

#include "sampler.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"

namespace futaba {

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

} // namespace futaba
