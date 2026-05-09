#pragma once

#include "types.cuh"
#include<cmath>

namespace futaba {

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

// -----------------------------------------------------------------------------
// COLOR UTILS (Replaces Nori's color.h)
// -----------------------------------------------------------------------------

// Calculate relative luminance of a linear RGB color
HD float getLuminance(const Color3f& c) {
    return c.x * 0.212671f + c.y * 0.715160f + c.z * 0.072169f;
}

// Convert linear RGB to sRGB (for displaying on monitors)
HD Color3f toSRGB(const Color3f& c) {
    Color3f result;
    for (int i = 0; i < 3; ++i) {
        float value = (i == 0) ? c.x : ((i == 1) ? c.y : c.z);
        if (value <= 0.0031308f) {
            value = 12.92f * value;
        } else {
            value = 1.055f * powf(value, 1.0f / 2.4f) - 0.055f;
        }
        if (i == 0) result.x = value;
        else if (i == 1) result.y = value;
        else result.z = value;
    }
    return result;
}

// -----------------------------------------------------------------------------
// GEOMETRY UTILS
// -----------------------------------------------------------------------------

// Complete a vector 'a' to form an orthonormal basis (a, b, c)
HD void coordinateSystem(const Vector3f& a, Vector3f& b, Vector3f& c) {
    if (fabsf(a.x) > fabsf(a.y)) {
        float invLen = FAST_RSQRT(a.x * a.x + a.z * a.z);
        c = Vector3f(a.z * invLen, 0.0f, -a.x * invLen);
    } else {
        float invLen = FAST_RSQRT(a.y * a.y + a.z * a.z);
        c = Vector3f(0.0f, a.z * invLen, -a.y * invLen);
    }
    b = cross(c, a);
}

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
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

// -----------------------------------------------------------------------------
// OPTICS
// -----------------------------------------------------------------------------

// Calculate unpolarized fresnel reflection coefficient
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

} // End namespace futaba