#pragma once

#include "types.cuh"

FUTABA_NAMESPACE_BEGIN

struct PhaseSample {
    Vector3f wi;     // Sampled incoming direction in world space
    float    pdf = 0.f;
    Color3f  weight = Color3f(1.f);
};

FUTABA_NAMESPACE_END
