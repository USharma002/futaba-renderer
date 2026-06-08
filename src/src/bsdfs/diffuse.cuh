#pragma once

#include "types.cuh"
#include "common.cuh"
#include "warp.cuh"
#include "frame.cuh"
#include "material.cuh"
#include "bsdf_sample.cuh"

namespace futaba {

struct Diffuse {
    Color3f albedo;

    HD Diffuse() : albedo(0.5f) {}
    HD Diffuse(const Color3f& a) : albedo(a) {}

    // f(wo, wi) = albedo / pi
    HD Color3f eval(const BSDFSample& bs) const {
        if (Frame::cos_theta(bs.wo) <= 0.f || Frame::cos_theta(bs.wi) <= 0.f)
            return Color3f(0.f);
        return albedo * INV_PI;
    }

    HD float pdf(const BSDFSample& bs) const {
        if (Frame::cos_theta(bs.wo) <= 0.f || Frame::cos_theta(bs.wi) <= 0.f)
            return 0.f;
        return Warp::squareToCosineHemispherePdf(bs.wo);
    }

    // Cosine-weighted hemisphere sample. weight = albedo (eval * cos / pdf simplifies).
    HD Color3f sample(BSDFSample& bs, const Point2f& s2) const {
        bs.wo           = Warp::squareToCosineHemisphere(s2);
        bs.pdf          = Warp::squareToCosineHemispherePdf(bs.wo);
        bs.weight       = albedo;
        bs.eta          = 1.f;
        bs.sampled_type = BSDF_ID_DIFFUSE;
        // front_face is set by prepare_bsdf() before this call; do not overwrite.
        return albedo;
    }
};

} // namespace futaba
