#pragma once
#include "scene.cuh"
#include "surface_interaction.cuh"
#include "emitter_sample.cuh"

namespace futaba {

// ---------------------------------------------------------------------------
// PowerEmitterSampler
// 
// Currently a placeholder that delegates to UniformEmitterSampler.
// To implement a custom light sampler (e.g., power-weighted sampling):
// 1. Define custom internal sampling logic here or in a new header.
// 2. To register a new sampler type:
//    - Add the new sampler struct.
//    - Add a variant to LightSamplerType in src/gui/launch_params.h.
//    - Add a selection option to the ComboBox in src/gui/gui.cpp.
//    - Add corresponding __raygen__ kernel(s) in src/gui/renderer_device.cu
//      instantiating Path<YourSampler> / VolumetricPath<YourSampler>.
//    - Register the new raygen program group(s) and SBT record(s) in
//      src/gui/renderer_host.cpp, updating the index selection in launch_render.
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
