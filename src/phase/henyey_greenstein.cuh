#pragma once

#include "types.cuh"
#include "common.cuh"
#include "frame.cuh"
#include "warp.cuh"

namespace futaba {

struct HenyeyGreensteinPhaseFunction {
    float g;

    HD HenyeyGreensteinPhaseFunction() : g(0.f) {}
    HD HenyeyGreensteinPhaseFunction(float g) : g(g) {}

    HD float eval(float cosTheta) const {
        if (fabsf(g) < 1e-4f) {
            return 0.25f / M_PI; // Isotropic phase function (1 / 4pi)
        }
        float g2 = g * g;
        float term = 1.f + g2 - 2.f * g * cosTheta;
        return 0.25f * (1.f - g2) / (M_PI * term * sqrtf(term));
    }

    HD void sample(const Vector3f& wi, const Point2f& u, Vector3f& wo, float& pdf) const {
        if (fabsf(g) < 1e-4f) {
            wo = Warp::squareToUniformSphere(u);
            pdf = Warp::squareToUniformSpherePdf(wo);
            return;
        }
        float cosTheta;
        float g2 = g * g;
        if (fabsf(g) < 1e-3f) {
            cosTheta = 1.f - 2.f * u.x;
        } else {
            float sqrVal = (1.f - g2) / (1.f + g * (u.x * 2.f - 1.f));
            cosTheta = (1.f + g2 - sqrVal * sqrVal) / (2.f * g);
        }
        float sinTheta = sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));
        float phi = 2.f * M_PI * u.y;
        
        Frame frame;
        frame.setFromNormal(wi);
        
        Vector3f wo_local(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
        wo = frame.to_world(wo_local);
        pdf = eval(cosTheta);
    }
};

} // namespace futaba
