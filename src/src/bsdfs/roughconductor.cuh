#pragma once

#include "bsdf_sample.cuh"
#include "microfacet.cuh"

namespace futaba {

struct RoughConductor {
    Color3f albedo;
    Color3f specularScale;
    Color3f eta;
    Color3f k;
    float   alpha;
    float   extIOR;

    HD RoughConductor()
        : albedo(0.f), specularScale(1.f), eta(0.f), k(1.f), alpha(0.1f), extIOR(1.000277f) {}

    HD RoughConductor(const Color3f& diffuseAlbedo,
                      float roughness,
                      float extIor,
                      const Color3f& conductorEta,
                      const Color3f& conductorK,
                      const Color3f& conductorSpecular)
        : albedo(diffuseAlbedo), specularScale(conductorSpecular), eta(conductorEta), k(conductorK),
          alpha(fmaxf(roughness, 1e-4f)), extIOR(extIor) {}

    HD Color3f eval(const BSDFSample& bs) const {
        Microfacet bsdf(albedo, alpha, extIOR, 1.f, true, eta, k, specularScale);
        return bsdf.eval(bs);
    }

    HD float pdf(const BSDFSample& bs) const {
        Microfacet bsdf(albedo, alpha, extIOR, 1.f, true, eta, k, specularScale);
        return bsdf.pdf(bs);
    }

    HD Color3f sample(BSDFSample& bs, const Point2f& s2) const {
        Microfacet bsdf(albedo, alpha, extIOR, 1.f, true, eta, k, specularScale);
        const Color3f result = bsdf.sample(bs, s2);
        bs.sampled_type = BSDF_ID_ROUGHCONDUCTOR;
        return result;
    }
};

} // namespace futaba
