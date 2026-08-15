#pragma once
 
#include "types.cuh"
#include "common.cuh"
#include "warp.cuh"
#include "frame.cuh"
#include "material.cuh"
#include "bsdf_sample.cuh"

FUTABA_NAMESPACE_BEGIN

struct Diffuse {
    Color3f albedo;

    HD Diffuse() : albedo(0.5f) {}
    HD explicit Diffuse(const Color3f& a) : albedo(a) {}

    HD Color3f eval(const BSDFSample& bs) const {
        if (Frame::cos_theta(bs.wo) <= 0.f || Frame::cos_theta(bs.wi) <= 0.f)
            return Color3f(0.f);
        return albedo * INV_PI;
    }

    HD float pdf(const BSDFSample& bs) const {
        if (Frame::cos_theta(bs.wo) <= 0.f || Frame::cos_theta(bs.wi) <= 0.f)
            return 0.f;
        return Warp::squareToCosineHemispherePdf(bs.wo);
    }

    HD void eval_pdf(const BSDFSample& bs, Color3f& f_out, float& pdf_out) const {
        const float cos_o = Frame::cos_theta(bs.wo);
        if (cos_o <= 0.f || Frame::cos_theta(bs.wi) <= 0.f) {
            f_out = Color3f(0.f);
            pdf_out = 0.f;
            return;
        }
        f_out = albedo * INV_PI;
        pdf_out = cos_o * INV_PI;
    }

    HD Color3f sample(BSDFSample& bs, const Point2f& u) const {
        if (Frame::cos_theta(bs.wi) <= 0.f) {
            bs.pdf = 0.f;
            bs.weight = Color3f(0.f);
            return Color3f(0.f);
        }
        bs.wo           = Warp::squareToCosineHemisphere(u);
        bs.pdf          = bs.wo.z * INV_PI;
        bs.weight       = albedo;
        bs.eta          = 1.f;
        bs.sampled_type = BSDF_ID_DIFFUSE;
        return albedo;
    }
};

FUTABA_NAMESPACE_END

