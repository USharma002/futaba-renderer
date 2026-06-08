#pragma once
#include "scene.cuh"
#include "surface_interaction.cuh"
#include "emitter_sample.cuh"

namespace futaba {

// ---------------------------------------------------------------------------
// PowerEmitterSampler
// 
// Placeholder light sampler for power-weighted light sampling.
// ---------------------------------------------------------------------------
struct PowerEmitterSampler {

    HD bool sample(const Scene& scene,
                   const SurfaceIntersection& ref,
                   const Point3f& u,
                   EmitterSample& es) const
    {
        // TODO: Implement power-weighted light sampling here.
        // Returning false indicates that light sampling is not implemented.
        return false;
    }

    HD float pdf(const Scene& scene,
                 int emitter_primitive_id,
                 const Vector3f& wi,
                 float dist) const
    {
        // TODO: Implement corresponding PDF evaluation here.
        return 0.f;
    }
};

} // namespace futaba
