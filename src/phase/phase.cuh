#pragma once

#include "phase_sample.cuh"
#include "hg.cuh"
#include "isotropic.cuh"

FUTABA_NAMESPACE_BEGIN

struct Phase {
    HD static float eval(float g, const Vector3f& wo, const Vector3f& wi) {
        return HenyeyGreenstein(g).eval(wo, wi);
    }

    HD static float pdf(float g, const Vector3f& wo, const Vector3f& wi) {
        return HenyeyGreenstein(g).pdf(wo, wi);
    }

    HD static Vector3f sample(float g, const Vector3f& wo, const Point2f& u, PhaseSample& ps) {
        return HenyeyGreenstein(g).sample(wo, u, ps);
    }
};

FUTABA_NAMESPACE_END
