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

        // return Color3f(si.t);
    }
};

} // namespace futaba
