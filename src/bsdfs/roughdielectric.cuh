#pragma once

#include "bsdf_sample.cuh"
#include "warp.cuh"
#include "frame.cuh"
#include "common.cuh"

namespace futaba {

struct RoughDielectric {
    Color3f albedo;
    float   alpha;
    float   extIOR;
    float   intIOR;

    HD RoughDielectric()
        : albedo(1.f), alpha(0.1f), extIOR(1.000277f), intIOR(1.5f) {}

    HD RoughDielectric(const Color3f& tint, float roughness, float extIor, float intIor)
        : albedo(tint), alpha(fmaxf(roughness, 1e-4f)), extIOR(extIor), intIOR(intIor) {}

    // G1 that works for both reflection (v.z > 0) and transmission (v.z < 0) lobes.
    // smithBeckmannG1 requires v.z > 0, so for the transmission wo (below the surface)
    // we flip both v and wh into the upper hemisphere together.
    HD static float G1(const Vector3f& v, const Vector3f& wh, float alpha) {
        const float sign = (v.z < 0.f) ? -1.f : 1.f;
        return Warp::smithBeckmannG1(v * sign, wh, alpha);
    }

    HD Color3f eval(const BSDFSample& bs) const {
        const float cosThetaI = Frame::cos_theta(bs.wi);
        const float cosThetaO = Frame::cos_theta(bs.wo);
        if (cosThetaI <= 0.f) return Color3f(0.f);

        const bool reflect = cosThetaO > 0.f;
        float etaI = bs.front_face ? extIOR : intIOR;
        float etaT = bs.front_face ? intIOR : extIOR;

        Vector3f wh;
        if (reflect) {
            if (cosThetaO <= 0.f) return Color3f(0.f);
            wh = normalize(bs.wi + bs.wo);
        } else {
            if (cosThetaO >= 0.f) return Color3f(0.f);
            wh = normalize(bs.wi * etaI + bs.wo * etaT);
            if (Frame::cos_theta(wh) < 0.f) wh = -wh;
        }

        const float whDotWi = dot(wh, bs.wi);
        if (whDotWi <= 0.f) return Color3f(0.f);

        const float D = Warp::beckmannD(wh, alpha);
        if (D <= 0.f) return Color3f(0.f);

        const float F = fresnel(whDotWi, etaI, etaT);
        const float G = G1(bs.wi, wh, alpha) * G1(bs.wo, wh, alpha);

        if (reflect) {
            return albedo * Color3f(F * D * G / fmaxf(4.f * cosThetaI * cosThetaO, 1e-8f));
        } else {
            const float whDotWo = dot(wh, bs.wo);
            const float denom   = etaI * whDotWi + etaT * whDotWo;
            const float factor  = (etaI * etaI * fabsf(whDotWi) * fabsf(whDotWo)) /
                                  fmaxf(denom * denom * cosThetaI * fabsf(cosThetaO), 1e-8f);
            return albedo * Color3f((1.f - F) * D * G * factor);
        }
    }

    HD float pdf(const BSDFSample& bs) const {
        const float cosThetaI = Frame::cos_theta(bs.wi);
        const float cosThetaO = Frame::cos_theta(bs.wo);
        if (cosThetaI <= 0.f) return 0.f;

        const bool reflect = cosThetaO > 0.f;
        float etaI = bs.front_face ? extIOR : intIOR;
        float etaT = bs.front_face ? intIOR : extIOR;

        Vector3f wh;
        if (reflect) {
            if (cosThetaO <= 0.f) return 0.f;
            wh = normalize(bs.wi + bs.wo);
        } else {
            if (cosThetaO >= 0.f) return 0.f;
            wh = normalize(bs.wi * etaI + bs.wo * etaT);
            if (Frame::cos_theta(wh) < 0.f) wh = -wh;
        }

        const float whDotWi = dot(wh, bs.wi);
        if (whDotWi <= 0.f) return 0.f;

        const float F   = fresnel(whDotWi, etaI, etaT);
        const float pWh = Warp::squareToBeckmannPdf(wh, alpha);

        if (reflect) {
            return F * pWh / fmaxf(4.f * whDotWi, 1e-8f);
        } else {
            const float whDotWo = dot(wh, bs.wo);
            const float denom   = etaI * whDotWi + etaT * whDotWo;
            const float dwh_dwo = (etaT * etaT * fabsf(whDotWo)) / fmaxf(denom * denom, 1e-8f);
            return (1.f - F) * pWh * dwh_dwo;
        }
    }

    HD Color3f sample(BSDFSample& bs, const Point2f& s2) const {
        const float cosThetaI = Frame::cos_theta(bs.wi);
        if (cosThetaI <= 0.f) {
            bs.pdf = 0.f; bs.weight = Color3f(0.f); return Color3f(0.f);
        }

        const Vector3f wh     = Warp::squareToBeckmann(s2, alpha);
        const float whDotWi   = dot(bs.wi, wh);
        if (whDotWi <= 0.f) {
            bs.pdf = 0.f; bs.weight = Color3f(0.f); return Color3f(0.f);
        }

        float etaI     = bs.front_face ? extIOR : intIOR;
        float etaT     = bs.front_face ? intIOR : extIOR;
        const float F  = fresnel(whDotWi, etaI, etaT);
        float eta      = etaI / etaT;

        const float sinT2 = eta * eta * fmaxf(0.f, 1.f - whDotWi * whDotWi);
        const bool  tir   = (sinT2 >= 1.f);

        if (tir || F >= 1.f - 1e-6f) {
            // Reflect (includes TIR)
            bs.wo = -bs.wi + 2.f * whDotWi * wh;
            if (Frame::cos_theta(bs.wo) <= 0.f) {
                bs.pdf = 0.f; bs.weight = Color3f(0.f); return Color3f(0.f);
            }
            bs.pdf          = Warp::squareToBeckmannPdf(wh, alpha) / fmaxf(4.f * whDotWi, 1e-8f);
            bs.eta          = 1.f;
            bs.sampled_type = BSDF_ID_ROUGHDIELECTRIC;
        } else {
            // Refract
            const float cosT = sqrtf(fmaxf(0.f, 1.f - sinT2));
            bs.wo = normalize(bs.wi * (-eta) + wh * (whDotWi * eta - cosT));
            if (Frame::cos_theta(bs.wo) >= 0.f) {
                bs.pdf = 0.f; bs.weight = Color3f(0.f); return Color3f(0.f);
            }
            const float whDotWo = dot(wh, bs.wo);
            const float denom   = etaI * whDotWi + etaT * whDotWo;
            const float dwh_dwo = (etaT * etaT * fabsf(whDotWo)) / fmaxf(denom * denom, 1e-8f);
            bs.pdf          = (1.f - F) * Warp::squareToBeckmannPdf(wh, alpha) * dwh_dwo;
            bs.eta          = eta;
            bs.sampled_type = BSDF_ID_ROUGHDIELECTRIC;
        }

        if (bs.pdf <= 0.f) {
            bs.weight = Color3f(0.f); return Color3f(0.f);
        }

        const float cosO = fabsf(Frame::cos_theta(bs.wo));
        bs.weight = eval(bs) * (cosO / bs.pdf);
        return bs.weight;
    }
};

} // namespace futaba