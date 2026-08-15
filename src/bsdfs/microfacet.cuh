#pragma once

#include "types.cuh"
#include "common.cuh"
#include "warp.cuh"
#include "frame.cuh"
#include "material.cuh"
#include "bsdf_sample.cuh"

FUTABA_NAMESPACE_BEGIN

struct Microfacet {
    Color3f kd;
    Color3f specularScale;
    Color3f eta;
    Color3f k;
    float   alpha;
    float   extIOR;
    float   intIOR;
    float   ks;
    bool    isConductor;

    HD Microfacet()
        : kd(0.5f), specularScale(1.f), eta(0.f), k(1.f), alpha(0.1f),
          extIOR(1.000277f), intIOR(1.5046f), ks(0.5f), isConductor(false) {}

    HD Microfacet(const Color3f& diffuseAlbedo,
                  float roughness,
                  float extIor,
                  float intIor,
                  bool conductor,
                  const Color3f& conductorEta,
                  const Color3f& conductorK,
                  const Color3f& conductorSpecular)
        : kd(diffuseAlbedo), specularScale(conductorSpecular), eta(conductorEta), k(conductorK),
          alpha(fmaxf(roughness, 1e-4f)), extIOR(extIor), intIOR(intIor), isConductor(conductor)
    {
        if (isConductor) {
            kd = Color3f(0.f);
            ks = 1.f;
        } else {
            // Specular *sampling* probability only (mixture weight between the
            // cosine-hemisphere and Beckmann sampling strategies). This is a
            // heuristic, not a physical energy split, so it must never be used
            // to scale the actual specular reflectance computed in eval() -
            // doing so would make brighter diffuse materials incorrectly dim
            // their specular highlight. Floored so the specular lobe stays
            // reachable via BSDF sampling even for near-white diffuse albedo.
            const float kdMax = fmaxf(kd.x, fmaxf(kd.y, kd.z));
            ks = clamp(1.f - kdMax, 0.05f, 1.f);
        }
    }

    HD Color3f fresnel_conductor(float cosThetaI) const {
        return fresnelConductor(cosThetaI, eta, k, extIOR, specularScale);
    }

    HD Color3f eval(const BSDFSample& bs) const {
        const float cosThetaI = Frame::cos_theta(bs.wi);
        const float cosThetaO = Frame::cos_theta(bs.wo);
        if (cosThetaI <= 0.f || cosThetaO <= 0.f)
            return Color3f(0.f);

        Color3f result = kd * INV_PI;

        const Vector3f hsum = bs.wi + bs.wo;
        if (hsum.lengthSquared() <= 1e-12f)
            return result;

        const Vector3f wh = normalize(hsum);
        const float D = Warp::beckmannD(wh, alpha);
        if (D <= 0.f)
            return result;

        const float cosWhWi = dot(wh, bs.wi);
        const float G = Warp::smithBeckmannG1(bs.wi, wh, alpha) * Warp::smithBeckmannG1(bs.wo, wh, alpha);
        // NOTE: no `ks` factor here - ks is a sampling-strategy mixture weight
        // (see constructor), not a physical reflectance scale. The specular
        // lobe's true magnitude is D * G * F / (4 cosI cosO) regardless of it.
        const float common = D * G / fmaxf(4.f * cosThetaI * cosThetaO, 1e-8f);

        if (isConductor) {
            const Color3f F = fresnel_conductor(cosWhWi);
            return result + F * common;
        }

        const float F = fresnel(cosWhWi, extIOR, intIOR);
        return result + Color3f(common * F);
    }

    HD float pdf(const BSDFSample& bs) const {
        const float cosThetaI = Frame::cos_theta(bs.wi);
        const float cosThetaO = Frame::cos_theta(bs.wo);
        if (cosThetaI <= 0.f || cosThetaO <= 0.f)
            return 0.f;

        const float diffusePdf = (1.f - ks) * Warp::squareToCosineHemispherePdf(bs.wo);

        float specPdf = 0.f;
        const Vector3f hsum = bs.wi + bs.wo;
        if (hsum.lengthSquared() > 1e-12f && ks > 0.f) {
            const Vector3f wh = normalize(hsum);
            const float cosThetaH = Frame::cos_theta(wh);
            const float woDotWh = dot(bs.wo, wh);
            if (cosThetaH > 0.f && woDotWh > 0.f) {
                const float pWh = Warp::squareToBeckmannPdf(wh, alpha);
                const float jacobian = 1.f / fmaxf(4.f * woDotWh, 1e-8f);
                specPdf = ks * pWh * jacobian;
            }
        }

        return diffusePdf + specPdf;
    }

    HD void eval_pdf(const BSDFSample& bs, Color3f& f_out, float& pdf_out) const {
        const float cosThetaI = Frame::cos_theta(bs.wi);
        const float cosThetaO = Frame::cos_theta(bs.wo);
        if (cosThetaI <= 0.f || cosThetaO <= 0.f) {
            f_out   = Color3f(0.f);
            pdf_out = 0.f;
            return;
        }

        f_out = kd * INV_PI;
        pdf_out = (1.f - ks) * (cosThetaO * INV_PI);

        const Vector3f hsum = bs.wi + bs.wo;
        if (hsum.lengthSquared() > 1e-12f) {
            const Vector3f wh = normalize(hsum);
            const float D = Warp::beckmannD(wh, alpha);
            const float cosThetaH = Frame::cos_theta(wh);
            const float woDotWh = dot(bs.wo, wh);

            if (D > 0.f && cosThetaH > 0.f && woDotWh > 0.f) {
                const float cosWhWi = dot(wh, bs.wi);
                const float G = Warp::smithBeckmannG1(bs.wi, wh, alpha) * Warp::smithBeckmannG1(bs.wo, wh, alpha);
                const float common = D * G / fmaxf(4.f * cosThetaI * cosThetaO, 1e-8f);

                if (isConductor) {
                    const Color3f F = fresnel_conductor(cosWhWi);
                    f_out += F * common;
                } else {
                    const float F = fresnel(cosWhWi, extIOR, intIOR);
                    f_out += Color3f(common * F);
                }

                if (ks > 0.f) {
                    const float pWh = D * cosThetaH;
                    const float jacobian = 1.f / fmaxf(4.f * woDotWh, 1e-8f);
                    pdf_out += ks * pWh * jacobian;
                }
            }
        }
    }

    HD Color3f sample(BSDFSample& bs, const Point2f& s2) const {
        if (Frame::cos_theta(bs.wi) <= 0.f) {
            bs.pdf = 0.f;
            bs.weight = Color3f(0.f);
            return Color3f(0.f);
        }

        const float u = clamp(s2.x, 0.f, 1.f - 1e-7f);

        if (u < ks && ks > 0.f) {
            const float uRemap = u / ks;
            const Vector3f wh = Warp::squareToBeckmann(Point2f(uRemap, s2.y), alpha);

            const float wiDotWh = dot(bs.wi, wh);
            bs.wo = -bs.wi + 2.f * wiDotWh * wh;

            if (Frame::cos_theta(bs.wo) <= 0.f || wiDotWh <= 0.f) {
                bs.pdf = 0.f;
                bs.weight = Color3f(0.f);
                return Color3f(0.f);
            }
        } else {
            const float diffuseProb = 1.f - ks;
            if (diffuseProb <= 0.f) {
                bs.pdf = 0.f;
                bs.weight = Color3f(0.f);
                return Color3f(0.f);
            }

            const float uRemap = (u - ks) / diffuseProb;
            bs.wo = Warp::squareToCosineHemisphere(Point2f(uRemap, s2.y));
        }

        Color3f f;
        eval_pdf(bs, f, bs.pdf);
        if (bs.pdf <= 0.f) {
            bs.weight = Color3f(0.f);
            return Color3f(0.f);
        }

        bs.weight       = f * (Frame::cos_theta(bs.wo) / bs.pdf);
        bs.eta          = 1.f;
        bs.sampled_type = BSDF_ID_MICROFACET;
        return bs.weight;
    }
};

FUTABA_NAMESPACE_END
