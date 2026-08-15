#pragma once

#include "types.cuh"
#include "launch_params.h"

FUTABA_NAMESPACE_BEGIN

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
    return (linear * (a * linear + Color3f(b))) / (linear * (c * linear + Color3f(d)) + Color3f(e));
}

// Reinhard tone mapping (local variant)
HD Color3f reinhard(const Color3f& linear) {
    return linear / (Color3f(1.f) + linear);
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

#include "common.cuh"

// Apply tone mapping based on mode
HD Color3f apply(const Color3f& linear, int mode) {
    switch (mode) {
        case TONEMAPPING_ACES:
            return aces(linear);
        case TONEMAPPING_REINHARD:
            return reinhard(linear);
        case TONEMAPPING_FILMIC:
            return filmic_uncharted2(linear);
        case TONEMAPPING_NONE:
        default:
            return none(linear);
    }
}

// Convert linear radiance to tonemapped sRGB 8-bit RGBA pixel
HD uchar4 pack_to_uchar4(const Color3f& linear_color, int tonemapping_mode) {
    Color3f final_color = toSRGB(apply(linear_color, tonemapping_mode));
    return make_uchar4(
        (unsigned char)clamp(final_color.x * 255.f, 0.f, 255.f),
        (unsigned char)clamp(final_color.y * 255.f, 0.f, 255.f),
        (unsigned char)clamp(final_color.z * 255.f, 0.f, 255.f),
        255
    );
}

} // namespace tonemap

FUTABA_NAMESPACE_END