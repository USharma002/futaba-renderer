#pragma once

#include "types.cuh"

// Result of sampling a BSDF at a surface point.
// wi is set by the caller (world -> local transform), wo is filled by the BSDF sample().
struct BSDFSample {
    Vector3f wo;          // Sampled outgoing direction (local frame)
    Vector3f wi;          // Incoming direction (local frame), set before calling sample()
    bool     front_face;  // True if hit the front/outside face
    float    pdf;         // Sampling PDF
    float    eta;         // IOR ratio (etaI / etaT), 1.0 for non-transmissive
    int      sampled_type;// BSDF_ID_* of the lobe that was sampled
    Color3f  weight;      // Throughput contribution: f * cos_theta / pdf

    HD BSDFSample()
        : wo(0.f), wi(0.f), front_face(true), pdf(0.f),
          eta(1.f), sampled_type(0), weight(0.f) {}

    HD bool is_valid() const { return pdf > 0.f; }
};
