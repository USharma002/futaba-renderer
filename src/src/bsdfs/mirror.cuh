#pragma once

#include "types.cuh"
#include "common.cuh"
#include "material.cuh"
#include "bsdf_sample.cuh"

namespace futaba {

struct Mirror {
    Color3f albedo;

    HD Mirror() : albedo(1.f) {}
    HD Mirror(const Color3f& a) : albedo(a) {}

    // Delta BSDF - eval and pdf are 0 everywhere; only sample() is meaningful.
    HD Color3f eval(const BSDFSample& /*bs*/) const { return Color3f(0.f); }
    HD float   pdf (const BSDFSample& /*bs*/) const { return 0.f; }

    // Perfect specular reflection: wo = reflect(wi) about z-axis in local frame.
    HD Color3f sample(BSDFSample& bs, const Point2f& /*s2*/) const {
        bs.wo           = Vector3f(-bs.wi.x, -bs.wi.y, bs.wi.z);
        bs.pdf          = 1.f;
        bs.weight       = albedo;
        bs.eta          = 1.f;
        bs.sampled_type = BSDF_ID_MIRROR;
        return albedo;
    }
};

} // namespace futaba
