#pragma once

#include "types.cuh"
#include "ray.cuh"

namespace futaba {

// Emitter: Light source abstraction (Mitsuba pattern)
struct Emitter {
    enum Type {
        AREA_LIGHT = 0,
        DIRECTIONAL_LIGHT = 1,
        POINT_LIGHT = 2
    };
    
    Type type;
    Color3f radiance;
    
    HD Emitter(Type t = AREA_LIGHT, Color3f L = Color3f(0.f))
        : type(t), radiance(L) {}
    
    // TODO: Sample illumination direction
    // HD Vector3f sampleDirection(Point2f u, float& pdf) const { ... }
    
    // TODO: Evaluate emitter radiance
    // HD Color3f eval(Vector3f wi) const { ... }
};

} // namespace futaba
