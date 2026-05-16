#pragma once

#include "types.cuh"
#include "launch_params.h"

namespace futaba {
namespace tonemap {

// No tone mapping, just return linear
HD Color3f none(const Color3f& linear) {
    return linear;
}

// ACES filmic tone mapping (approximation)
HD Color3f aces(const Color3f& linear) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;

    // Element-wise operations: (x * (a*x + b)) / (x * (c*x + d) + e)
    Color3f ax_b = Color3f(
        linear.x * a,
        linear.y * a,
        linear.z * a
    ) + Color3f(b);
    
    Color3f num = Color3f(
        linear.x * ax_b.x,
        linear.y * ax_b.y,
        linear.z * ax_b.z
    );

    Color3f cx_d = Color3f(
        linear.x * c,
        linear.y * c,
        linear.z * c
    ) + Color3f(d);
    
    Color3f denom = Color3f(
        linear.x * cx_d.x,
        linear.y * cx_d.y,
        linear.z * cx_d.z
    ) + Color3f(e);

    Color3f result = Color3f(
        num.x / denom.x,
        num.y / denom.y,
        num.z / denom.z
    );
    return result;
}

// Reinhardt tone mapping (local variant)
HD Color3f reinhardt(const Color3f& linear) {
    // result = x / (1 + x)
    Color3f denom = Color3f(
        1.f + linear.x,
        1.f + linear.y,
        1.f + linear.z
    );
    Color3f result = Color3f(
        linear.x / denom.x,
        linear.y / denom.y,
        linear.z / denom.z
    );
    return result;
}

// Filmic tone mapping (Uncharted 2 style)
HD Color3f filmic_uncharted2(const Color3f& x) {
    const float A = 0.15f;
    const float B = 0.50f;
    const float C = 0.10f;
    const float D = 0.20f;
    const float E = 0.02f;
    const float F = 0.30f;
    const float W = 11.2f;  // White point

    // Define tone curve as inline helper
    auto tonecurve = [A, B, C, D, E, F](const float v) -> float {
        float num = v * (A * v + C * B) + D * E;
        float denom = v * (A * v + B) + D * F;
        return num / denom - E / F;
    };

    Color3f result;
    result.x = tonecurve(x.x);
    result.y = tonecurve(x.y);
    result.z = tonecurve(x.z);

    float white_scale = 1.0f / tonecurve(W);
    result = result * white_scale;
    return result;
}

// Apply tone mapping based on mode
HD Color3f apply(const Color3f& linear, int mode) {
    switch (mode) {
        case TONEMAPPING_ACES:
            return aces(linear);
        case TONEMAPPING_REINHARDT:
            return reinhardt(linear);
        case TONEMAPPING_FILMIC:
            return filmic_uncharted2(linear);
        case TONEMAPPING_NONE:
        default:
            return none(linear);
    }
}

} // namespace tonemap
} // namespace futaba
