#pragma once

#include "types.cuh"
#include "common.cuh"
#include "warp.cuh"
#include "frame.cuh"
#include "material.cuh"
#include "bsdf_sample.cuh"

namespace futaba {

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
            const float kdMax = fmaxf(kd.x, fmaxf(kd.y, kd.z));
            ks = 1.f - clamp(kdMax, 0.f, 1.f);
        }
    }

    HD static float fresnel_conductor_channel(float cosThetaI, float eta, float k) {
        const float cosI = fmaxf(cosThetaI, 0.f);
        const float cos2 = cosI * cosI;
        const float sin2 = fmaxf(0.f, 1.f - cos2);

        const float eta2 = eta * eta;
        const float k2   = k * k;

        const float t0 = eta2 - k2 - sin2;
        const float a2pb2 = sqrtf(t0 * t0 + 4.f * eta2 * k2);
        const float a = sqrtf(fmaxf(0.f, 0.5f * (a2pb2 + t0)));

        const float t1 = a2pb2 + cos2;
        const float t2 = 2.f * cosI * a;
        const float Rs = (t1 - t2) / fmaxf(t1 + t2, 1e-8f);

        const float t3 = cos2 * a2pb2 + sin2 * sin2;
        const float t4 = t2 * sin2;
        const float Rp = Rs * (t3 - t4) / fmaxf(t3 + t4, 1e-8f);

        return clamp(0.5f * (Rs + Rp), 0.f, 1.f);
    }

    HD Color3f fresnel_conductor(float cosThetaI) const {
        const float etaScale = fmaxf(extIOR, 1e-6f);
        const Color3f relEta = eta / etaScale;
        const Color3f relK   = k   / etaScale;
        return Color3f(
            fresnel_conductor_channel(cosThetaI, relEta.x, relK.x),
            fresnel_conductor_channel(cosThetaI, relEta.y, relK.y),
            fresnel_conductor_channel(cosThetaI, relEta.z, relK.z)
        ) * specularScale;
    }

    HD Color3f eval(const BSDFSample& bs) const {
        const float cosThetaI = Frame::cos_theta(bs.wi);
        const float cosThetaO = Frame::cos_theta(bs.wo);
        if (cosThetaI <= 0.f || cosThetaO <= 0.f)
            return Color3f(0.f);

        Color3f result = kd * INV_PI;

        const Vector3f hsum = bs.wi + bs.wo;
        if (hsum.lengthSquared() <= 1e-12f || ks <= 0.f)
            return result;

        const Vector3f wh = normalize(hsum);
        const float D = Warp::beckmannD(wh, alpha);
        if (D <= 0.f)
            return result;

        const float cosWhWi = dot(wh, bs.wi);
        const float G = Warp::smithBeckmannG1(bs.wi, wh, alpha) * Warp::smithBeckmannG1(bs.wo, wh, alpha);
        const float common = ks * D * G / fmaxf(4.f * cosThetaI * cosThetaO, 1e-8f);

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

        bs.pdf = pdf(bs);
        if (bs.pdf <= 0.f) {
            bs.weight = Color3f(0.f);
            return Color3f(0.f);
        }

        bs.weight = eval(bs) * (Frame::cos_theta(bs.wo) / bs.pdf);
        bs.eta          = 1.f;
        bs.sampled_type = BSDF_ID_MICROFACET;
        return bs.weight;
    }
};

} // namespace futaba
