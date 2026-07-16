#pragma once

#include "types.cuh"
#include "common.cuh"
#include "frame.cuh"
#include "bsdf_sample.cuh"
#include "material.cuh"

FUTABA_NAMESPACE_BEGIN

struct ThinDielectric {
    Color3f albedo;
    float   intIOR;

    HD ThinDielectric() : albedo(1.f), intIOR(1.5f) {}
    HD ThinDielectric(const Color3f& tint, float ior) : albedo(tint), intIOR(ior) {}

    HD Color3f eval(const BSDFSample& /*bs*/) const { return Color3f(0.f); }
    HD float   pdf (const BSDFSample& /*bs*/) const { return 0.f; }

    HD Color3f sample(BSDFSample& bs, const Point2f& s2) const {
        float etaI = bs.front_face ? 1.f : intIOR;
        float etaT = bs.front_face ? intIOR : 1.f;
        float cosI = Frame::cos_theta(bs.wi);
        float Fr   = fresnel(cosI, etaI, etaT);

        // Account for internal reflections inside the thin slab
        Fr *= 2.f / (1.f + Fr);

        // Reflect
        if (s2.x < Fr) {
            bs.wo           = Vector3f(-bs.wi.x, -bs.wi.y, bs.wi.z);
            bs.pdf          = fmaxf(Fr, 1e-6f);
            bs.weight       = albedo;
            bs.eta          = 1.f;
            bs.sampled_type = BSDF_ID_THINDIELECTRIC;
            return bs.weight;
        }

        // Transmit (straight through without bending)
        bs.wo           = -bs.wi;
        bs.pdf          = fmaxf(1.f - Fr, 1e-6f);
        bs.weight       = albedo;
        bs.eta          = 1.f;
        bs.sampled_type = BSDF_ID_THINDIELECTRIC;
        return bs.weight;
    }
};

FUTABA_NAMESPACE_END
