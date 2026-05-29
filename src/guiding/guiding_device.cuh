#pragma once

#include "types.cuh"
#include "common.cuh"
#include "warp.cuh"
#include "frame.cuh"
#include "bsdf_sample.cuh"
#include "guiding.h"
#include "surface_interaction.cuh"

namespace futaba {

struct GuidingDistribution {
    int mode;
    Point3f p;
    Normal3f n;
    Frame frame;

    HD GuidingDistribution(int m, const SurfaceIntersection& si)
        : mode(m), p(si.p), n(si.n), frame(si.frame) {}

    // Evaluate the guiding distribution value (e.g. represented as incoming radiance or a product)
    HD Color3f eval(const BSDFSample& bs) const {
        if (mode == PATH_GUIDING_NONE) {
            return Color3f(0.f);
        }
        if (mode == PATH_GUIDING_SD_TREE) {
            // TODO: Query SD-Tree representation at position `p` for direction `bs.wo`
            return Color3f(1.f); // placeholder
        }
        if (mode == PATH_GUIDING_NPM) {
            // TODO: Query NPM representation at position `p` for direction `bs.wo`
            return Color3f(1.f); // placeholder
        }
        return Color3f(0.f);
    }

    // Evaluate the solid-angle PDF of the guiding distribution for a direction
    HD float pdf(const BSDFSample& bs) const {
        if (mode == PATH_GUIDING_NONE) {
            return 0.f;
        }
        if (mode == PATH_GUIDING_SD_TREE) {
            // TODO: Evaluate SD-Tree PDF at position `p` for direction `bs.wo`
            // Fallback: Cosine-weighted or uniform hemisphere PDF
            if (Frame::cos_theta(bs.wo) <= 0.f) return 0.f;
            return Warp::squareToCosineHemispherePdf(bs.wo);
        }
        if (mode == PATH_GUIDING_NPM) {
            // TODO: Evaluate NPM PDF at position `p` for direction `bs.wo`
            if (Frame::cos_theta(bs.wo) <= 0.f) return 0.f;
            return Warp::squareToCosineHemispherePdf(bs.wo);
        }
        return 0.f;
    }

    // Sample a direction from the guiding distribution.
    // Matches the BSDF::sample interface exactly.
    HD Color3f sample(BSDFSample& bs, const Point2f& s2) const {
        if (mode == PATH_GUIDING_NONE) {
            bs.pdf = 0.f;
            bs.weight = Color3f(0.f);
            return Color3f(0.f);
        }
        if (mode == PATH_GUIDING_SD_TREE) {
            // TODO: Sample direction wo from SD-Tree at position `p`
            // Fallback: cosine hemisphere sample
            bs.wo = Warp::squareToCosineHemisphere(s2);
            bs.pdf = Warp::squareToCosineHemispherePdf(bs.wo);
            bs.weight = Color3f(1.f);
            bs.eta = 1.f;
            bs.sampled_type = BSDF_ID_DIFFUSE; // placeholder or custom ID
            return bs.weight;
        }
        if (mode == PATH_GUIDING_NPM) {
            // TODO: Sample direction wo from NPM at position `p`
            bs.wo = Warp::squareToCosineHemisphere(s2);
            bs.pdf = Warp::squareToCosineHemispherePdf(bs.wo);
            bs.weight = Color3f(1.f);
            bs.eta = 1.f;
            bs.sampled_type = BSDF_ID_DIFFUSE;
            return bs.weight;
        }
        return Color3f(0.f);
    }
};

} // namespace futaba
