#pragma once

#include "types.cuh"
#include "common.cuh"
#include "warp.cuh"

namespace futaba {

struct IsotropicPhaseFunction {
    HD float eval(float /*cosTheta*/) const {
        return INV_FOURPI;
    }

    HD void sample(const Vector3f& /*wi*/, const Point2f& u, Vector3f& wo, float& pdf) const {
        wo = Warp::squareToUniformSphere(u);
        pdf = Warp::squareToUniformSpherePdf(wo);
    }
};

} // namespace futaba
