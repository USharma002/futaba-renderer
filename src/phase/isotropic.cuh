#pragma once

#include "types.cuh"
#include "common.cuh"

namespace futaba {

struct IsotropicPhaseFunction {
    HD float eval(float /*cosTheta*/) const {
        return 0.25f / M_PI;
    }

    HD void sample(const Vector3f& /*wi*/, const Point2f& u, Vector3f& wo, float& pdf) const {
        float z = 1.f - 2.f * u.x;
        float r = sqrtf(fmaxf(0.f, 1.f - z * z));
        float phi = 2.f * M_PI * u.y;
        wo = Vector3f(r * cosf(phi), r * sinf(phi), z);
        pdf = 0.25f / M_PI;
    }
};

} // namespace futaba
