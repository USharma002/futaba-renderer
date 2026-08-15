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

// Result of an emitter connection / sampling query (corresponds to Mitsuba 3's DirectionSample3f)
struct DirectionSample {
    Point3f  p            = Point3f(0.f);  // Sampled point on emitter (12 bytes)
    Vector3f d            = Vector3f(0.f);  // Unit direction (12 bytes)
    Color3f  Le           = Color3f(0.f);   // Radiance emitted (12 bytes)
    Color3f  weight       = Color3f(0.f);   // Throughput weight: Le / pdf (12 bytes)
    float    dist         = 0.f;            // Distance to sampled point (4 bytes)
    float    pdf          = 0.f;            // Solid-angle PDF (4 bytes)
    int      primitive_id = -1;             // Global triangle index (4 bytes)
    int      mesh_id      = -1;             // Mesh ID (4 bytes)
    bool     delta        = false;          // True for Dirac delta emitters (1 byte)
    uint8_t  _pad[3]      = {0, 0, 0};      // Alignment padding (3 bytes)

    HD bool is_valid() const { return pdf > 0.f && isfinite(pdf); }
};

// Alias for seamless backwards compatibility
using EmitterSample = DirectionSample;

FUTABA_NAMESPACE_END

