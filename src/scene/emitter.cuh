#pragma once

#include <cstdint>
#include "types.cuh"

FUTABA_NAMESPACE_BEGIN

enum class EmitterType : uint32_t {
    None        = 0,
    Area        = 1,
    Point       = 2,
    Directional = 3,
};

struct EmitterInstance {
    EmitterType type      = EmitterType::None;
    Color3f     radiance  = Color3f(0.f);
    Point3f     position  = Point3f(0.f, 0.f, 0.f);   // point emitters
    Vector3f    direction = Vector3f(0.f, -1.f, 0.f); // directional emitters
    bool        twoSided  = true;                     // area emitters: emit from both faces unless true

    EmitterInstance() = default;
    explicit EmitterInstance(EmitterType t) : type(t) {}
};

FUTABA_NAMESPACE_END