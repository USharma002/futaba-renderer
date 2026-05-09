#pragma once

#include "types.cuh"
#include "common.cuh"
#include "frame.cuh"
#include "bsdf_sample.cuh"
#include "material.cuh"

namespace futaba {

struct Dielectric {
    Color3f albedo;
    float   intIOR;

    HD Dielectric() : albedo(1.f), intIOR(1.5f) {}
    HD Dielectric(const Color3f& tint, float ior) : albedo(tint), intIOR(ior) {}

    HD Color3f sample(BSDFSample& bs, const Point2f& s2) const {
        float etaI = bs.front_face ? 1.f : intIOR;
        float etaT = bs.front_face ? intIOR : 1.f;
        float cosI = Frame::cos_theta(bs.wi);
        float Fr   = fresnel(cosI, etaI, etaT);

        // Reflect (also handles TIR via the Fr >= 1 branch)
        if (Fr >= 1.f - 1e-6f || s2.x < Fr) {
            bs.wo           = Vector3f(-bs.wi.x, -bs.wi.y, bs.wi.z);
            bs.pdf          = fmaxf(Fr, 1e-6f);
            bs.weight       = albedo;
            bs.eta          = 1.f;
            bs.sampled_type = BSDF_ID_DIELECTRIC;
            return bs.weight;
        }

        // Check for total internal reflection
        float eta       = etaI / etaT;
        float sin2I     = fmaxf(0.f, 1.f - cosI * cosI);
        float sin2T     = eta * eta * sin2I;
        if (sin2T >= 1.f) {
            bs.wo           = Vector3f(-bs.wi.x, -bs.wi.y, bs.wi.z);
            bs.pdf          = 1.f;
            bs.weight       = albedo;
            bs.eta          = 1.f;
            bs.sampled_type = BSDF_ID_DIELECTRIC;
            return bs.weight;
        }

        // Refract. The (etaI/etaT)**2 factor is the correct radiance-transport correction.
        float cosT      = sqrtf(fmaxf(0.f, 1.f - sin2T));
        bs.wo           = Vector3f(-eta * bs.wi.x, -eta * bs.wi.y, -cosT);
        bs.pdf          = fmaxf(1.f - Fr, 1e-6f);
        bs.weight       = albedo * (eta * eta);
        bs.eta          = eta;
        bs.sampled_type = BSDF_ID_DIELECTRIC;
        return bs.weight;
    }
};

} // namespace futaba
