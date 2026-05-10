#pragma once

#include "bsdf_sample.cuh"
#include "dielectric.cuh"

namespace futaba {

struct RoughDielectric {
    Color3f albedo;
    float   alpha;
    float   extIOR;
    float   intIOR;

    HD RoughDielectric()
        : albedo(1.f), alpha(0.1f), extIOR(1.000277f), intIOR(1.5f) {}

    HD RoughDielectric(const Color3f& tint,
                       float roughness,
                       float extIor,
                       float intIor)
        : albedo(tint), alpha(fmaxf(roughness, 1e-4f)), extIOR(extIor), intIOR(intIor) {}

    HD Color3f eval(const BSDFSample& /*bs*/) const { return Color3f(0.f); }

    HD float pdf(const BSDFSample& /*bs*/) const { return 0.f; }

    HD Color3f sample(BSDFSample& bs, const Point2f& s2) const {
        Dielectric bsdf(albedo, intIOR);
        (void)alpha;
        (void)extIOR;
        const Color3f result = bsdf.sample(bs, s2);
        bs.sampled_type = BSDF_ID_ROUGHDIELECTRIC;
        return result;
    }
};

} // namespace futaba
