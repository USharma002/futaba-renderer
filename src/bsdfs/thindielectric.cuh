#pragma once

#include "types.cuh"
#include "common.cuh"
#include "frame.cuh"
#include "bsdf_sample.cuh"
#include "material.cuh"

FUTABA_NAMESPACE_BEGIN

struct ThinDielectric {
    Color3f albedo;
    float   extIOR;
    float   intIOR;

    HD ThinDielectric() : albedo(1.f), extIOR(1.000277f), intIOR(1.5f) {}
    HD explicit ThinDielectric(const Color3f& tint, float int_ior = 1.5f, float ext_ior = 1.000277f)
        : albedo(tint), extIOR(ext_ior), intIOR(int_ior) {}

    // Dirac Delta BSDF: continuous eval and pdf are 0
    HD Color3f eval(const BSDFSample& /*bs*/) const { return Color3f(0.f); }
    HD float   pdf (const BSDFSample& /*bs*/) const { return 0.f; }

    HD void eval_pdf(const BSDFSample& /*bs*/, Color3f& f_out, float& pdf_out) const {
        f_out   = Color3f(0.f);
        pdf_out = 0.f;
    }

    HD Color3f sample(BSDFSample& bs, const Point2f& u) const {
        float cosI = fabsf(Frame::cos_theta(bs.wi));
        float Fr   = fresnel(cosI, extIOR, intIOR);

        // Account for internal bounces in a thin slab: Fr_thin = 2 Fr / (1 + Fr)
        Fr = (Fr < 1.f) ? (2.f * Fr / (1.f + Fr)) : 1.f;

        if (u.x < Fr) {
            // Reflect
            bs.wo           = Vector3f(-bs.wi.x, -bs.wi.y, bs.wi.z);
            bs.pdf          = fmaxf(Fr, 1e-6f);
            bs.weight       = albedo;
            bs.eta          = 1.f;
            bs.sampled_type = BSDF_ID_THINDIELECTRIC;
            return bs.weight;
        }

        // Transmit without deviation
        bs.wo           = -bs.wi;
        bs.pdf          = fmaxf(1.f - Fr, 1e-6f);
        bs.weight       = albedo;
        bs.eta          = 1.f;
        bs.sampled_type = BSDF_ID_THINDIELECTRIC;
        return bs.weight;
    }
};

FUTABA_NAMESPACE_END

