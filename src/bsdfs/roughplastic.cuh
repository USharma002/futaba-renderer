#pragma once

#include "types.cuh"
#include "common.cuh"
#include "warp.cuh"
#include "frame.cuh"
#include "material.cuh"
#include "bsdf_sample.cuh"

FUTABA_NAMESPACE_BEGIN

// Physically accurate rough plastic model: dielectric microfacet coating over a Lambertian diffuse substrate
// with internal multiple reflection energy compensation (Kelemen & Szirmay-Kalos / Walter et al.)
struct RoughPlastic {
    Color3f albedo;
    float   alpha;
    float   extIOR;
    float   intIOR;

    HD RoughPlastic()
        : albedo(0.5f), alpha(0.1f), extIOR(1.000277f), intIOR(1.5046f) {}

    HD RoughPlastic(const Color3f& diffuseAlbedo,
                    float roughness,
                    float extIor,
                    float intIor)
        : albedo(diffuseAlbedo), alpha(fmaxf(roughness, 1e-4f)), extIOR(extIor), intIOR(intIor) {}

    HD Color3f eval(const BSDFSample& bs) const {
        const float cosThetaI = Frame::cos_theta(bs.wi);
        const float cosThetaO = Frame::cos_theta(bs.wo);
        if (cosThetaI <= 0.f || cosThetaO <= 0.f)
            return Color3f(0.f);

        // Specular microfacet reflection
        Color3f spec(0.f);
        const Vector3f hsum = bs.wi + bs.wo;
        if (hsum.lengthSquared() > 1e-12f) {
            const Vector3f wh = normalize(hsum);
            const float cosThetaH = Frame::cos_theta(wh);
            const float cosWhWi   = dot(wh, bs.wi);

            if (cosThetaH > 0.f && cosWhWi > 0.f) {
                const float D = Warp::beckmannD(wh, alpha);
                if (D > 0.f) {
                    const float G = Warp::smithBeckmannG1(bs.wi, wh, alpha) * Warp::smithBeckmannG1(bs.wo, wh, alpha);
                    const float F = fresnel(cosWhWi, extIOR, intIOR);
                    spec = Color3f(F * D * G / fmaxf(4.f * cosThetaI * cosThetaO, 1e-8f));
                }
            }
        }

        // Diffuse substrate scattering modulated by Fresnel transmission
        const float Fi = fresnel(cosThetaI, extIOR, intIOR);
        const float Fo = fresnel(cosThetaO, extIOR, intIOR);
        const float eta = intIOR / extIOR;
        const float Fdr = fresnelDiffuseReflectance(eta);
        const float denom = fmaxf(1.f - Fdr, 1e-4f);
        const float factor = (1.f - Fi) * (1.f - Fo) / (denom * (eta * eta));

        Color3f diff = albedo * (factor * INV_PI);

        return spec + diff;
    }

    HD float pdf(const BSDFSample& bs) const {
        const float cosThetaI = Frame::cos_theta(bs.wi);
        const float cosThetaO = Frame::cos_theta(bs.wo);
        if (cosThetaI <= 0.f || cosThetaO <= 0.f)
            return 0.f;

        const float Fi = fresnel(cosThetaI, extIOR, intIOR);
        const float ks = clamp(Fi, 0.1f, 0.9f);

        float specPdf = 0.f;
        const Vector3f hsum = bs.wi + bs.wo;
        if (hsum.lengthSquared() > 1e-12f) {
            const Vector3f wh = normalize(hsum);
            const float cosThetaH = Frame::cos_theta(wh);
            const float woDotWh = dot(bs.wo, wh);
            if (cosThetaH > 0.f && woDotWh > 0.f) {
                const float pWh = Warp::squareToBeckmannPdf(wh, alpha);
                specPdf = pWh / fmaxf(4.f * woDotWh, 1e-8f);
            }
        }

        const float diffPdf = cosThetaO * INV_PI;
        return ks * specPdf + (1.f - ks) * diffPdf;
    }

    HD void eval_pdf(const BSDFSample& bs, Color3f& f_out, float& pdf_out) const {
        const float cosThetaI = Frame::cos_theta(bs.wi);
        const float cosThetaO = Frame::cos_theta(bs.wo);
        if (cosThetaI <= 0.f || cosThetaO <= 0.f) {
            f_out   = Color3f(0.f);
            pdf_out = 0.f;
            return;
        }

        const float Fi = fresnel(cosThetaI, extIOR, intIOR);
        const float Fo = fresnel(cosThetaO, extIOR, intIOR);
        const float eta = intIOR / extIOR;
        const float Fdr = fresnelDiffuseReflectance(eta);
        const float denom = fmaxf(1.f - Fdr, 1e-4f);
        const float factor = (1.f - Fi) * (1.f - Fo) / (denom * (eta * eta));

        f_out = albedo * (factor * INV_PI);

        const float ks = clamp(Fi, 0.1f, 0.9f);
        pdf_out = (1.f - ks) * (cosThetaO * INV_PI);

        const Vector3f hsum = bs.wi + bs.wo;
        if (hsum.lengthSquared() > 1e-12f) {
            const Vector3f wh = normalize(hsum);
            const float cosThetaH = Frame::cos_theta(wh);
            const float woDotWh = dot(bs.wo, wh);

            if (cosThetaH > 0.f && woDotWh > 0.f) {
                const float D = Warp::beckmannD(wh, alpha);
                if (D > 0.f) {
                    const float cosWhWi = dot(wh, bs.wi);
                    const float G = Warp::smithBeckmannG1(bs.wi, wh, alpha) * Warp::smithBeckmannG1(bs.wo, wh, alpha);
                    const float F = fresnel(cosWhWi, extIOR, intIOR);
                    f_out += Color3f(F * D * G / fmaxf(4.f * cosThetaI * cosThetaO, 1e-8f));

                    const float pWh = D * cosThetaH;
                    pdf_out += ks * (pWh / fmaxf(4.f * woDotWh, 1e-8f));
                }
            }
        }
    }

    HD Color3f sample(BSDFSample& bs, const Point2f& s2) const {
        const float cosThetaI = Frame::cos_theta(bs.wi);
        if (cosThetaI <= 0.f) {
            bs.pdf = 0.f;
            bs.weight = Color3f(0.f);
            return Color3f(0.f);
        }

        const float Fi = fresnel(cosThetaI, extIOR, intIOR);
        const float ks = clamp(Fi, 0.1f, 0.9f);

        if (s2.x < ks) {
            // Sample specular lobe
            const Point2f uRemap(s2.x / ks, s2.y);
            const Vector3f wh = Warp::squareToBeckmann(uRemap, alpha);
            const float wiDotWh = dot(bs.wi, wh);
            if (wiDotWh <= 0.f) {
                bs.pdf = 0.f; bs.weight = Color3f(0.f); return Color3f(0.f);
            }

            bs.wo = -bs.wi + 2.f * wiDotWh * wh;
            if (Frame::cos_theta(bs.wo) <= 0.f) {
                bs.pdf = 0.f; bs.weight = Color3f(0.f); return Color3f(0.f);
            }
        } else {
            // Sample diffuse substrate
            const Point2f uRemap((s2.x - ks) / (1.f - ks), s2.y);
            bs.wo = Warp::squareToCosineHemisphere(uRemap);
        }

        Color3f f;
        eval_pdf(bs, f, bs.pdf);
        if (bs.pdf <= 0.f) {
            bs.weight = Color3f(0.f);
            return Color3f(0.f);
        }

        bs.weight       = f * (Frame::cos_theta(bs.wo) / bs.pdf);
        bs.eta          = 1.f;
        bs.sampled_type = BSDF_ID_ROUGHPLASTIC;
        return bs.weight;
    }
};

FUTABA_NAMESPACE_END
