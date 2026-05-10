#pragma once

#include "types.cuh"
#include "common.cuh"
#include "frame.cuh"
#include <cmath>

// Warp functions for importance sampling
// All points have [0, 1]^d uniform samples
struct Warp {

    HD static Point2f squareToUniformSquare(const Point2f &sample2){
        return sample2;
    }

    HD static float squareToUniformSquarePdf(const Point2f &sample2){
        if ((sample2.x >= 0 && sample2.y >= 0) && (sample2.x <= 1 && sample2.y <= 1)){
            return 1.0f;
        }

        return 0.0f;
    }

    HD static Vector3f squareToUniformSphere(const Point3f& sample2){
        float phi = 2.0f * M_PI * sample2.x;
        float cos_theta = 1.0f - 2.0f * sample2.y;
        float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
        
        return Vector3f(
            sin_theta * cosf(phi),
            sin_theta * sinf(phi),
            cos_theta
        );
    }

    HD static float squareToUniformSpherePdf(const Vector3f&) {
        return INV_PI / 4.0f;
    }

    // Uniform hemisphere sampling
    HD static Vector3f squareToUniformHemisphere(const Point2f& sample2) {
        float phi = 2.0f * M_PI * sample2.x;
        float cos_theta = 1.0f - sample2.y;
        float sin_theta = sqrtf(sample2.y * (2.0f - sample2.y));
        
        return Vector3f(
            sin_theta * cosf(phi),
            sin_theta * sinf(phi),
            cos_theta
        );
    }
    
    // Probability density for uniform hemisphere sampling
    HD static float squareToUniformHemispherePdf(const Vector3f&) {
        return INV_PI / 2.0f;
    }


    // Uniformly sample a vector on the unit hemisphere around the pole (0,0,1) with respect to projected solid angles
    HD static Vector3f squareToCosineHemisphere(const Point2f& sample2) {
        float phi = 2.0f * M_PI * sample2.x;
        float cos_theta = sqrtf(sample2.y);
        float sin_theta = sqrtf(1.0f - sample2.y);
        
        return Vector3f(
            sin_theta * cosf(phi),
            sin_theta * sinf(phi),
            cos_theta
        );
    }
    
    // Probability density of squareToCosineHemisphere()
    HD static float squareToCosineHemispherePdf(const Vector3f& wo) {
        float cos_theta = wo.z;
        if (cos_theta <= 0.0f) return 0.0f;
        return cos_theta * INV_PI;
    }

    // Beckmann normal distribution function D(wh).
    HD static float beckmannD(const Vector3f& wh, float alpha) {
        const float cosThetaH = futaba::Frame::cos_theta(wh);
        if (cosThetaH <= 0.f)
            return 0.f;

        const float a = fmaxf(alpha, 1e-4f);
        const float cos2 = cosThetaH * cosThetaH;
        const float sin2 = fmaxf(0.f, 1.f - cos2);
        const float tan2 = sin2 / fmaxf(cos2, 1e-8f);
        const float a2 = a * a;
        return expf(-tan2 / a2) / (M_PI * a2 * cos2 * cos2);
    }

    // PDF of sampling a microfacet normal wh from Beckmann: p(wh) = D(wh) * cos(theta_h).
    HD static float squareToBeckmannPdf(const Vector3f& wh, float alpha) {
        const float cosThetaH = futaba::Frame::cos_theta(wh);
        if (cosThetaH <= 0.f)
            return 0.f;
        return beckmannD(wh, alpha) * cosThetaH;
    }

    // Sample a microfacet normal from Beckmann distribution.
    // sample2.x controls theta_h via inversion, sample2.y controls phi.
    HD static Vector3f squareToBeckmann(const Point2f& sample2, float alpha) {
        const float a = fmaxf(alpha, 1e-4f);
        const float u = clamp(sample2.x, 0.f, 1.f - 1e-7f);
        const float v = clamp(sample2.y, 0.f, 1.f - 1e-7f);

        const float phi = 2.f * M_PI * v;
        const float tan2Theta = -a * a * logf(fmaxf(1.f - u, 1e-7f));
        const float cosTheta = 1.f / sqrtf(1.f + tan2Theta);
        const float sinTheta = sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));

        return Vector3f(
            sinTheta * cosf(phi),
            sinTheta * sinf(phi),
            cosTheta
        );
    }

    // Smith G1 masking term approximation for Beckmann roughness.
    HD static float smithBeckmannG1(const Vector3f& v, const Vector3f& wh, float alpha) {
        const float cosThetaV = futaba::Frame::cos_theta(v);
        if (cosThetaV <= 0.f)
            return 0.f;

        const float vh = dot(v, wh);
        if (vh / cosThetaV <= 0.f)
            return 0.f;

        const float sin2 = fmaxf(0.f, 1.f - cosThetaV * cosThetaV);
        if (sin2 <= 1e-8f)
            return 1.f;

        const float tanTheta = sqrtf(sin2) / fmaxf(cosThetaV, 1e-8f);
        if (tanTheta <= 1e-8f)
            return 1.f;

        const float a = fmaxf(alpha, 1e-4f);
        const float b = 1.f / (a * tanTheta);
        if (b >= 1.6f)
            return 1.f;

        const float b2 = b * b;
        return (3.535f * b + 2.181f * b2) / (1.f + 2.276f * b + 2.577f * b2);
    }

};
