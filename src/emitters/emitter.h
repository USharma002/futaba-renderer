#pragma once

#include "../core/types.cuh"

namespace futaba {

struct SurfaceIntersection;

// Minimal CPU-side polymorphic emitter base for future extension.
struct Emitter {
    virtual ~Emitter() {}
    virtual Color3f eval(const SurfaceIntersection &si) const {
        (void)si;
        return Color3f(0.f);
    }
    // Traversal / parameter access / texture resolution hooks.
};

} // namespace futaba
