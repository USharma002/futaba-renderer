#pragma once

#include "homogeneous.cuh"

namespace futaba {

enum MediumType {
    MEDIUM_HOMOGENEOUS = 0
};

struct Medium {
    MediumType type;
    HomogeneousMedium homogeneous;

    HD Medium() : type(MEDIUM_HOMOGENEOUS), homogeneous() {}
    HD Medium(MediumType t, const HomogeneousMedium& h) : type(t), homogeneous(h) {}

    HD Color3f evalTransmittance(float dist) const {
        switch (type) {
            case MEDIUM_HOMOGENEOUS:
            default:
                return homogeneous.evalTransmittance(dist);
        }
    }

    HD float sampleChannel(float u_chan, float& sigma_t_c) const {
        switch (type) {
            case MEDIUM_HOMOGENEOUS:
            default:
                return homogeneous.sampleChannel(u_chan, sigma_t_c);
        }
    }

    HD const PhaseFunction& getPhaseFunction() const {
        return homogeneous.phase;
    }
};

} // namespace futaba
