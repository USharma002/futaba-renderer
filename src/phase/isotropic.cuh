#pragma once

#include "types.cuh"
#include "common.cuh"
#include "warp.cuh"
#include "phase_sample.cuh"

FUTABA_NAMESPACE_BEGIN

struct IsotropicPhase {
    HD IsotropicPhase() {}

    HD float eval(const Vector3f& /*wo*/, const Vector3f& /*wi*/) const {
        return Inv4Pi;
    }

    HD float pdf(const Vector3f& /*wo*/, const Vector3f& /*wi*/) const {
        return Inv4Pi;
    }

    HD Vector3f sample(const Vector3f& /*wo*/, const Point2f& u, PhaseSample& ps) const {
        ps.wi     = Warp::squareToUniformSphere(u);
        ps.pdf    = Inv4Pi;
        ps.weight = Color3f(1.0f);
        return ps.wi;
    }
};

FUTABA_NAMESPACE_END
