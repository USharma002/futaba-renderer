#pragma once

#include "types.cuh"
#include "material.cuh"

FUTABA_NAMESPACE_BEGIN

enum class TransportMode {
    Radiance   = 0,
    Importance = 1
};

struct BSDFContext {
    TransportMode mode = TransportMode::Radiance;
    HD BSDFContext(TransportMode m = TransportMode::Radiance) : mode(m) {}
};

struct BSDFSample {
    Vector3f wo;           // Sampled direction in local shading frame (12 bytes)
    Vector3f wi;           // Incident direction in local shading frame (12 bytes)
    Color3f  weight;       // f * |cos_theta(wo)| / pdf throughput weight (12 bytes)
    float    pdf;          // Solid-angle PDF of wo (4 bytes)
    float    eta;          // Relative index of refraction (etaI / etaT) (4 bytes)
    int      sampled_type; // BSDFType of the sampled lobe (4 bytes)
    bool     front_face;   // True if the ray hit the outside/front face (1 byte)
    uint8_t  _pad[3] = {0, 0, 0}; // Alignment padding to 4-byte boundary (3 bytes)

    HD BSDFSample()
        : wo(0.f), wi(0.f), weight(0.f), pdf(0.f),
          eta(1.f), sampled_type(BSDF_ID_DIFFUSE), front_face(true) {}

    HD bool is_valid() const { return pdf > 0.f && isfinite(pdf); }
    HD bool is_transmissive() const { return (wo.z * wi.z < 0.f); }
};

FUTABA_NAMESPACE_END

