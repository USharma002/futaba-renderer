#pragma once

#include "types.cuh"
#include <cmath>
#include <cstdio>
#include <limits>

FUTABA_NAMESPACE_BEGIN

constexpr float Infinity = std::numeric_limits<float>::infinity();

// -----------------------------------------------------------------------------
// Shared host-side CUDA error-checking helper. Every host .cpp/.cu file that
// allocates or copies CUDA memory should use this instead of rolling its own.
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t _err = (call);                                      \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_err));             \
        }                                                                      \
    } while (0)


// -----------------------------------------------------------------------------
// CONSTANTS (From Nori's common.h)
// -----------------------------------------------------------------------------
#ifndef M_PI
#define M_PI         3.14159265358979323846f
#define INV_PI       0.31830988618379067154f
#define INV_TWOPI    0.15915494309189533577f
#define INV_FOURPI   0.07957747154594766788f
#define SQRT_TWO     1.41421356237309504880f
#define INV_SQRT_TWO 0.70710678118654752440f
#endif

constexpr float Pi         = 3.14159265358979323846f;
constexpr float TwoPi      = 6.28318530717958647692f;
constexpr float InvPi      = 0.31830988618379067154f;
constexpr float Inv2Pi     = 0.15915494309189533577f;
constexpr float Inv4Pi     = 0.07957747154594766788f;

#define Epsilon 1e-4f



// -----------------------------------------------------------------------------
// ALIASES
// -----------------------------------------------------------------------------
// A Normal is mathematically just a Vector3, but we alias it for code clarity
typedef Vector3f Normal3f;

// -----------------------------------------------------------------------------
// GENERAL MATH UTILS
// -----------------------------------------------------------------------------

HD float degToRad(float deg) { return deg * (M_PI / 180.0f); }
HD float radToDeg(float rad) { return rad * (180.0f / M_PI); }

// Float modulo (Always positive)
HD float mod(float a, float b) {
    float r = fmodf(a, b);
    return (r < 0.0f) ? r + b : r;
}

// Power heuristic Multiple Importance Sampling (MIS) weight calculation.
HD float mis_weight(float pdf_a, float pdf_b) {
    const float a2 = pdf_a * pdf_a;
    const float b2 = pdf_b * pdf_b;
    const float sum = a2 + b2;
    return (sum > 0.f) ? (a2 / sum) : 0.f;
}

// -----------------------------------------------------------------------------
// COLOR UTILS (Replaces Nori's color.h)
// -----------------------------------------------------------------------------

// Calculate relative luminance of a linear RGB color
HD float getLuminance(const Color3f& c) {
    return c.x * 0.212671f + c.y * 0.715160f + c.z * 0.072169f;
}

// Convert linear RGB to sRGB (for displaying on monitors)
HD Color3f toSRGB(const Color3f& c) {
    auto srgb_channel = [](float v) {
        return (v <= 0.0031308f) ? (12.92f * v) : (1.055f * powf(v, 1.f / 2.4f) - 0.055f);
    };
    return Color3f(srgb_channel(c.x), srgb_channel(c.y), srgb_channel(c.z));
}

// -----------------------------------------------------------------------------
// GEOMETRY UTILS
// -----------------------------------------------------------------------------


// Spherical to Cartesian coordinates
HD Vector3f sphericalDirection(float theta, float phi) {
    float sinTheta, cosTheta, sinPhi, cosPhi;
    
#ifdef __CUDA_ARCH__
    // Hardware-accelerated single instruction on GPU
    sincosf(theta, &sinTheta, &cosTheta);
    sincosf(phi, &sinPhi, &cosPhi);
#else
    // Standard C++ fallback for CPU (MSVC)
    sinTheta = std::sin(theta);
    cosTheta = std::cos(theta);
    sinPhi   = std::sin(phi);
    cosPhi   = std::cos(phi);
#endif

    return Vector3f(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}


HD float length(const Vector3f& v) {
    return v.length();
}

// -----------------------------------------------------------------------------
// OPTICS
// -----------------------------------------------------------------------------

// Calculate unpolarized fresnel reflection coefficient for dielectrics
HD float fresnel(float cosThetaI, float extIOR, float intIOR) {
    float etaI = extIOR, etaT = intIOR;

    if (cosThetaI < 0.0f) {
        // Swap indices of refraction if hitting from the inside
        float temp = etaI;
        etaI = etaT;
        etaT = temp;
        cosThetaI = -cosThetaI;
    }

    // Snell's law
    float sinThetaI = sqrtf(FAST_MAX(0.0f, 1.0f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;

    // Total internal reflection
    if (sinThetaT >= 1.0f) return 1.0f;

    float cosThetaT = sqrtf(FAST_MAX(0.0f, 1.0f - sinThetaT * sinThetaT));

    float rs = (etaI * cosThetaI - etaT * cosThetaT) / (etaI * cosThetaI + etaT * cosThetaT);
    float rp = (etaT * cosThetaI - etaI * cosThetaT) / (etaT * cosThetaI + etaI * cosThetaT);

    return (rs * rs + rp * rp) * 0.5f;
}

// Conductor Fresnel for a single spectral channel (Born & Wolf / PBRT)
HD float fresnelConductorChannel(float cosThetaI, float eta, float k) {
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

// Spectral Conductor Fresnel with external IOR scaling
HD Color3f fresnelConductor(float cosThetaI, const Color3f& eta, const Color3f& k, float extIOR, const Color3f& specularScale = Color3f(1.f)) {
    if (eta.x == 0.f && eta.y == 0.f && eta.z == 0.f) {
        return specularScale;
    }
    const float etaScale = fmaxf(extIOR, 1e-6f);
    const Color3f relEta = eta / etaScale;
    const Color3f relK   = k   / etaScale;
    return Color3f(
        fresnelConductorChannel(cosThetaI, relEta.x, relK.x),
        fresnelConductorChannel(cosThetaI, relEta.y, relK.y),
        fresnelConductorChannel(cosThetaI, relEta.z, relK.z)
    ) * specularScale;
}

// Average internal diffuse reflectance for dielectric coating over diffuse substrate
HD float fresnelDiffuseReflectance(float eta) {
    if (eta < 1.0f) {
        // From inside to outside when eta < 1
        return -0.4399f + (0.7099f / eta) - (0.3319f / (eta * eta)) + (0.0636f / (eta * eta * eta));
    } else {
        return -1.4399f / (eta * eta) + (0.7099f / eta) + 0.6681f + 0.0636f * eta;
    }
}

FUTABA_NAMESPACE_END