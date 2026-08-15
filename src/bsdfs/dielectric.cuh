#pragma once

#include "types.cuh"
#include "common.cuh"
#include "frame.cuh"
#include "bsdf_sample.cuh"
#include "material.cuh"

FUTABA_NAMESPACE_BEGIN

struct Dielectric {
    Color3f albedo;
    float   extIOR;
    float   intIOR;

    HD Dielectric() : albedo(1.f), extIOR(1.000277f), intIOR(1.5f) {}
    HD explicit Dielectric(const Color3f& tint, float int_ior = 1.5f, float ext_ior = 1.000277f)
        : albedo(tint), extIOR(ext_ior), intIOR(int_ior) {}

    // Dirac Delta BSDF: continuous eval and pdf are 0
    HD Color3f eval(const BSDFSample& /*bs*/) const { return Color3f(0.f); }
    HD float   pdf (const BSDFSample& /*bs*/) const { return 0.f; }

    HD void eval_pdf(const BSDFSample& /*bs*/, Color3f& f_out, float& pdf_out) const {
        f_out   = Color3f(0.f);
        pdf_out = 0.f;
    }

    HD Color3f sample(BSDFSample& bs, const Point2f& u) const {
        float etaI = bs.front_face ? extIOR : intIOR;
        float etaT = bs.front_face ? intIOR : extIOR;
        float cosI = Frame::cos_theta(bs.wi);
        float Fr   = fresnel(cosI, etaI, etaT);

        // Reflect (also handles TIR via Fr == 1)
        if (Fr >= 1.f - 1e-6f || u.x < Fr) {
            bs.wo           = Vector3f(-bs.wi.x, -bs.wi.y, bs.wi.z);
            bs.pdf          = fmaxf(Fr, 1e-6f);
            bs.weight       = albedo;
            bs.eta          = 1.f;
            bs.sampled_type = BSDF_ID_DIELECTRIC;
            return bs.weight;
        }

        // Refract: (etaI/etaT)**2 factor is the radiance transport compression/expansion factor
        float eta       = etaI / etaT;
        float sin2I     = fmaxf(0.f, 1.f - cosI * cosI);
        float sin2T     = eta * eta * sin2I;
        float cosT      = sqrtf(fmaxf(0.f, 1.f - sin2T));
        bs.wo           = Vector3f(-eta * bs.wi.x, -eta * bs.wi.y, (cosI > 0.f ? -cosT : cosT));
        bs.pdf          = fmaxf(1.f - Fr, 1e-6f);
        bs.weight       = albedo * (eta * eta);
        bs.eta          = eta;
        bs.sampled_type = BSDF_ID_DIELECTRIC;
        return bs.weight;
    }
};

FUTABA_NAMESPACE_END

