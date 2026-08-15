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
        SurfaceInteraction si;
        scene.intersect(ray, ray.mint, ray.maxt, si);

        if (!si.is_valid())
            return Color3f(0.f);

        return Color3f(1.0f / (1.0f + 0.1f * si.t));
    }
};

FUTABA_NAMESPACE_END

