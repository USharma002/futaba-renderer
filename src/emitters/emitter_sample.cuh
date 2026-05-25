#pragma once
#include "types.cuh"

namespace futaba {
    // Result of one emitter draw for NEE.
    struct EmitterSample {
        Point3f  p;              // Sampled point on the emitter
        Vector3f d;              // Direction: shading point → emitter
        float    dist  = 0.f;    // Distance to the sampled point
        float    pdf   = 0.f;    // Solid-angle PDF
        Color3f  Le    = {0.f};  // Emitted radiance toward shading point
        bool     delta = false;  // True for point/directional emitters
        int      primitive_id = -1; // Global triangle index (area); -1 otherwise
        int      mesh_id      = -1; // Mesh owning the sampled triangle
    };
} // namespace futaba
