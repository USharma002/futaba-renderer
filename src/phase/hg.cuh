#pragma once

#include "types.cuh"
#include "common.cuh"
#include "frame.cuh"
#include "phase_sample.cuh"
#include <cmath>

FUTABA_NAMESPACE_BEGIN

struct HenyeyGreenstein {
    float g;

    HD explicit HenyeyGreenstein(float g_val = 0.f) : g(g_val) {}

    // Evaluate phase function p(wo, wi) = p(cosTheta)
    HD float eval(const Vector3f& wo, const Vector3f& wi) const {
        float cosTheta = dot(wo, wi);
        if (fabsf(g) < 1e-3f) {
            return Inv4Pi;
        }
        float sqr = 1.0f + g * g - 2.0f * g * cosTheta;
        if (sqr <= 0.f) return Inv4Pi;
        return Inv4Pi * (1.0f - g * g) / (sqr * sqrtf(sqr));
    }

    HD float pdf(const Vector3f& wo, const Vector3f& wi) const {
        return eval(wo, wi);
    }

    // Sample scattered direction wi given incoming direction wo
    HD Vector3f sample(const Vector3f& wo, const Point2f& u, PhaseSample& ps) const {
        float cosTheta;
        if (fabsf(g) < 1e-3f) {
            cosTheta = 1.0f - 2.0f * u.x;
        } else {
            float sqr = (1.0f - g * g) / (1.0f - g + 2.0f * g * u.x);
            cosTheta = (1.0f + g * g - sqr * sqr) / (2.0f * g);
        }
        cosTheta = fmaxf(-1.f, fminf(1.f, cosTheta));
        float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
        float phi = TwoPi * u.y;

        Vector3f localDir(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
        Frame frame(wo);
        ps.wi     = frame.to_world(localDir);
        ps.pdf    = eval(wo, ps.wi);
        ps.weight = Color3f(1.0f);
        return ps.wi;
    }
};

FUTABA_NAMESPACE_END
