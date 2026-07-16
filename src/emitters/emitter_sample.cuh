#pragma once
#include "types.cuh"

FUTABA_NAMESPACE_BEGIN
    // Data structure holding CDF tables/arrays for basic emitter sampling
    struct CDFLightSamplerData {
        float* emitterTriangleCdf = nullptr;
        int*   emissiveTriangleIndices = nullptr;
        int*   emissiveGlobalToIndex = nullptr;
        int    emissiveTriCount = 0;
        float  emitterTriangleFuncSum = 0.f;
    };

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
FUTABA_NAMESPACE_END
