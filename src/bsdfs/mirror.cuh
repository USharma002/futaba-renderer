#pragma once

#include "types.cuh"
#include "common.cuh"
#include "frame.cuh"
#include "material.cuh"
#include "bsdf_sample.cuh"

FUTABA_NAMESPACE_BEGIN

struct Mirror {
    Color3f albedo;

    HD Mirror() : albedo(1.f) {}
    HD explicit Mirror(const Color3f& a) : albedo(a) {}

    // Dirac Delta BSDF: continuous eval and pdf are 0
    HD Color3f eval(const BSDFSample& /*bs*/) const { return Color3f(0.f); }
    HD float   pdf (const BSDFSample& /*bs*/) const { return 0.f; }

    HD void eval_pdf(const BSDFSample& /*bs*/, Color3f& f_out, float& pdf_out) const {
        f_out   = Color3f(0.f);
        pdf_out = 0.f;
    }

    // Perfect specular reflection: wo = reflect(wi) about surface normal (z-axis)
    HD Color3f sample(BSDFSample& bs, const Point2f& /*u*/) const {
        if (Frame::cos_theta(bs.wi) <= 0.f) {
            bs.pdf = 0.f;
            bs.weight = Color3f(0.f);
            return Color3f(0.f);
        }
        bs.wo           = Vector3f(-bs.wi.x, -bs.wi.y, bs.wi.z);
        bs.pdf          = 1.f;
        bs.weight       = albedo;
        bs.eta          = 1.f;
        bs.sampled_type = BSDF_ID_MIRROR;
        return albedo;
    }
};

FUTABA_NAMESPACE_END

