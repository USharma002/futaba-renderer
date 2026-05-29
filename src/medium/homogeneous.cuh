#pragma once

#include "types.cuh"
#include "phase_function.cuh"

namespace futaba {

struct HomogeneousMedium {
    Color3f sigmaS;
    Color3f sigmaA;
    Color3f sigmaT;
    PhaseFunction phase;

    HD HomogeneousMedium() : sigmaS(0.f), sigmaA(0.f), sigmaT(0.f), phase() {}
    HD HomogeneousMedium(const Color3f& s, const Color3f& a, float g)
        : sigmaS(s), sigmaA(a), sigmaT(s + a), phase(PHASE_HENYEY_GREENSTEIN, g) {}

    HD Color3f evalTransmittance(float dist) const {
        return exp(-sigmaT * dist);
    }

    HD float sampleChannel(float u_chan, float& sigma_t_c) const {
        int channel = clamp((int)(u_chan * 3.f), 0, 2);
        sigma_t_c = sigmaT[channel];
        return sigma_t_c;
    }
};

} // namespace futaba
