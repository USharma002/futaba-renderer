#pragma once

#include "emitter.h"
#include "../core/types.cuh"
#include "../integrator/surface_interaction.cuh"

namespace futaba {

// CPU-side AreaEmitter: simple radiance-backed emitter for now.
struct AreaEmitter : public Emitter {
    Color3f radiance;

    AreaEmitter() : radiance(0.f) {}
    explicit AreaEmitter(const Color3f &r) : radiance(r) {}

    // Evaluate emission at a surface interaction (no textures yet)
    Color3f eval(const SurfaceIntersection &si) const override {
        (void)si;
        return radiance;
    }
};

} // namespace futaba
