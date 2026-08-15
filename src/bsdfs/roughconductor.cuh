#pragma once

#include "types.cuh"
#include "common.cuh"
#include "warp.cuh"
#include "frame.cuh"
#include "material.cuh"
#include "bsdf_sample.cuh"

FUTABA_NAMESPACE_BEGIN

// Physically accurate rough conductor microfacet BSDF (pure specular microfacet reflection)
struct RoughConductor {
    Color3f specularScale;
    Color3f eta;
    Color3f k;
    float   alpha;
    float   extIOR;

    HD RoughConductor()
        : specularScale(1.f), eta(0.f), k(1.f), alpha(0.1f), extIOR(1.000277f) {}

    HD RoughConductor(const Color3f& /*diffuseAlbedo*/,
                      float roughness,
                      float extIor,
                      const Color3f& conductorEta,
                      const Color3f& conductorK,
                      const Color3f& conductorSpecular)
        : specularScale(conductorSpecular), eta(conductorEta), k(conductorK),
          alpha(fmaxf(roughness, 1e-4f)), extIOR(extIor) {}

    HD Color3f eval(const BSDFSample& bs) const {
        const float cosThetaI = Frame::cos_theta(bs.wi);
        const float cosThetaO = Frame::cos_theta(bs.wo);
        if (cosThetaI <= 0.f || cosThetaO <= 0.f)
            return Color3f(0.f);

        const Vector3f hsum = bs.wi + bs.wo;
        if (hsum.lengthSquared() <= 1e-12f)
            return Color3f(0.f);

        const Vector3f wh = normalize(hsum);
        const float D = Warp::beckmannD(wh, alpha);
        if (D <= 0.f)
            return Color3f(0.f);

        const float cosWhWi = dot(wh, bs.wi);
        if (cosWhWi <= 0.f)
            return Color3f(0.f);

        const float G = Warp::smithBeckmannG1(bs.wi, wh, alpha) * Warp::smithBeckmannG1(bs.wo, wh, alpha);
        const Color3f F = fresnelConductor(cosWhWi, eta, k, extIOR, specularScale);

        return F * (D * G / fmaxf(4.f * cosThetaI * cosThetaO, 1e-8f));
    }

    HD float pdf(const BSDFSample& bs) const {
        const float cosThetaI = Frame::cos_theta(bs.wi);
        const float cosThetaO = Frame::cos_theta(bs.wo);
        if (cosThetaI <= 0.f || cosThetaO <= 0.f)
            return 0.f;

        const Vector3f hsum = bs.wi + bs.wo;
        if (hsum.lengthSquared() <= 1e-12f)
            return 0.f;

        const Vector3f wh = normalize(hsum);
        const float cosThetaH = Frame::cos_theta(wh);
        const float woDotWh = dot(bs.wo, wh);
        if (cosThetaH <= 0.f || woDotWh <= 0.f)
            return 0.f;

        const float D = Warp::beckmannD(wh, alpha);
        return (D * cosThetaH) / fmaxf(4.f * woDotWh, 1e-8f);
    }

    HD void eval_pdf(const BSDFSample& bs, Color3f& f_out, float& pdf_out) const {
        const float cosThetaI = Frame::cos_theta(bs.wi);
        const float cosThetaO = Frame::cos_theta(bs.wo);
        if (cosThetaI <= 0.f || cosThetaO <= 0.f) {
            f_out   = Color3f(0.f);
            pdf_out = 0.f;
            return;
        }

        const Vector3f hsum = bs.wi + bs.wo;
        if (hsum.lengthSquared() <= 1e-12f) {
            f_out   = Color3f(0.f);
            pdf_out = 0.f;
            return;
        }

        const Vector3f wh = normalize(hsum);
        const float D = Warp::beckmannD(wh, alpha);
        const float cosThetaH = Frame::cos_theta(wh);
        const float woDotWh = dot(bs.wo, wh);

        if (D <= 0.f || cosThetaH <= 0.f || woDotWh <= 0.f) {
            f_out   = Color3f(0.f);
            pdf_out = 0.f;
            return;
        }

        const float cosWhWi = dot(wh, bs.wi);
        const float G = Warp::smithBeckmannG1(bs.wi, wh, alpha) * Warp::smithBeckmannG1(bs.wo, wh, alpha);
        const Color3f F = fresnelConductor(cosWhWi, eta, k, extIOR, specularScale);

        f_out   = F * (D * G / fmaxf(4.f * cosThetaI * cosThetaO, 1e-8f));
        pdf_out = (D * cosThetaH) / fmaxf(4.f * woDotWh, 1e-8f);
    }

    HD Color3f sample(BSDFSample& bs, const Point2f& s2) const {
        const float cosThetaI = Frame::cos_theta(bs.wi);
        if (cosThetaI <= 0.f) {
            bs.pdf = 0.f;
            bs.weight = Color3f(0.f);
            return Color3f(0.f);
        }

        const Vector3f wh = Warp::squareToBeckmann(s2, alpha);
        const float wiDotWh = dot(bs.wi, wh);
        if (wiDotWh <= 0.f) {
            bs.pdf = 0.f;
            bs.weight = Color3f(0.f);
            return Color3f(0.f);
        }

        bs.wo = -bs.wi + 2.f * wiDotWh * wh;
        const float cosThetaO = Frame::cos_theta(bs.wo);
        if (cosThetaO <= 0.f) {
            bs.pdf = 0.f;
            bs.weight = Color3f(0.f);
            return Color3f(0.f);
        }

        const float cosThetaH = Frame::cos_theta(wh);
        const float D = Warp::beckmannD(wh, alpha);
        bs.pdf = (D * cosThetaH) / fmaxf(4.f * wiDotWh, 1e-8f);

        // Analytical microfacet weight: (f * cosO) / pdf
        // D and 4*cosO cancel out exactly, eliminating exponential underflow at low roughness:
        const float G = Warp::smithBeckmannG1(bs.wi, wh, alpha) * Warp::smithBeckmannG1(bs.wo, wh, alpha);
        const Color3f F = fresnelConductor(wiDotWh, eta, k, extIOR, specularScale);

        bs.weight       = F * (G * wiDotWh / fmaxf(cosThetaI * cosThetaH, 1e-8f));
        bs.eta          = 1.f;
        bs.sampled_type = BSDF_ID_ROUGHCONDUCTOR;
        return bs.weight;
    }
};

FUTABA_NAMESPACE_END
