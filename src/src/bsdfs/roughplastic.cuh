#pragma once

#include "bsdf_sample.cuh"
#include "microfacet.cuh"

namespace futaba {

struct RoughPlastic {
    Color3f albedo;
    float   alpha;
    float   extIOR;
    float   intIOR;

    HD RoughPlastic()
        : albedo(0.5f), alpha(0.1f), extIOR(1.000277f), intIOR(1.5046f) {}

    HD RoughPlastic(const Color3f& diffuseAlbedo,
                    float roughness,
                    float extIor,
                    float intIor)
        : albedo(diffuseAlbedo), alpha(fmaxf(roughness, 1e-4f)), extIOR(extIor), intIOR(intIor) {}

    HD Color3f eval(const BSDFSample& bs) const {
        Microfacet bsdf(albedo, alpha, extIOR, intIOR, false, Color3f(0.f), Color3f(1.f), Color3f(1.f));
        return bsdf.eval(bs);
    }

    HD float pdf(const BSDFSample& bs) const {
        Microfacet bsdf(albedo, alpha, extIOR, intIOR, false, Color3f(0.f), Color3f(1.f), Color3f(1.f));
        return bsdf.pdf(bs);
    }

    HD Color3f sample(BSDFSample& bs, const Point2f& s2) const {
        Microfacet bsdf(albedo, alpha, extIOR, intIOR, false, Color3f(0.f), Color3f(1.f), Color3f(1.f));
        const Color3f result = bsdf.sample(bs, s2);
        bs.sampled_type = BSDF_ID_ROUGHPLASTIC;
        return result;
    }
};

} // namespace futaba
