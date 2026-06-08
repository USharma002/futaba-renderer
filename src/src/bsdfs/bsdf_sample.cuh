#pragma once

#include "types.cuh"

// BSDFSample is the primary in/out record for BSDF sampling.
//
// Contract:
//  1. Caller fills `wi` (local-frame incoming direction) and `front_face`
//     via SurfaceIntersection::prepare_bsdf() before calling any BSDF method.
//  2. BSDF::sample() fills `wo`, `pdf`, `weight`, `eta`, `sampled_type`.
//  3. weight encodes  f(wi,wo) * |cos_theta(wo)| / pdf  - multiply path
//     throughput directly by weight; do NOT apply an extra cosine.

namespace futaba {

struct BSDFSample {
    Vector3f wo;           // Sampled outgoing direction (local frame)
    Vector3f wi;           // Incoming direction (local frame) - set by prepare_bsdf()
    bool     front_face;   // True if the ray hit the front/outside face
    float    pdf;          // Solid-angle PDF of wo
    float    eta;          // IOR ratio etaI/etaT; 1 for non-transmissive events
    int      sampled_type; // BSDFType of the lobe that was actually sampled
    Color3f  weight;       // f * |cos_theta(wo)| / pdf - apply to path throughput

    HD BSDFSample()
        : wo(0.f), wi(0.f), front_face(true), pdf(0.f),
          eta(1.f), sampled_type(BSDF_ID_DIFFUSE), weight(0.f) {}

    HD bool is_valid() const { return pdf > 0.f; }
};

} // namespace futaba
