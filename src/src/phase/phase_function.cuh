#pragma once

#include "isotropic.cuh"
#include "henyey_greenstein.cuh"

namespace futaba {

enum PhaseFunctionType {
    PHASE_ISOTROPIC = 0,
    PHASE_HENYEY_GREENSTEIN = 1
};

struct PhaseFunction {
    PhaseFunctionType type;
    float g;

    HD PhaseFunction() : type(PHASE_ISOTROPIC), g(0.f) {}
    HD PhaseFunction(PhaseFunctionType t, float g) : type(t), g(g) {}

    HD float eval(float cosTheta) const {
        switch (type) {
            case PHASE_HENYEY_GREENSTEIN: {
                HenyeyGreensteinPhaseFunction hg(g);
                return hg.eval(cosTheta);
            }
            case PHASE_ISOTROPIC:
            default: {
                IsotropicPhaseFunction iso;
                return iso.eval(cosTheta);
            }
        }
    }

    HD void sample(const Vector3f& wi, const Point2f& u, Vector3f& wo, float& pdf) const {
        switch (type) {
            case PHASE_HENYEY_GREENSTEIN: {
                HenyeyGreensteinPhaseFunction hg(g);
                hg.sample(wi, u, wo, pdf);
                break;
            }
            case PHASE_ISOTROPIC:
            default: {
                IsotropicPhaseFunction iso;
                iso.sample(wi, u, wo, pdf);
                break;
            }
        }
    }
};

} // namespace futaba
