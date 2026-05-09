#pragma once

#include "types.cuh"

namespace futaba {

enum BSDFType {
    BSDF_ID_DIFFUSE    = 0,
    BSDF_ID_DIELECTRIC = 1,
    BSDF_ID_MIRROR     = 2,
};

struct Material {
    Color3f  albedo;
    Color3f  emission;
    float    intIOR;
    BSDFType type;

    HD Material()
        : albedo(0.5f), emission(0.0f), intIOR(1.5f), type(BSDF_ID_DIFFUSE) {}

    HD Material(const Color3f& a, const Color3f& e = Color3f(0.0f))
        : albedo(a), emission(e), intIOR(1.5f), type(BSDF_ID_DIFFUSE) {}

    HD Material(const Color3f& a, const Color3f& e, BSDFType t, float ior)
        : albedo(a), emission(e), intIOR(ior), type(t) {}
};

} // namespace futaba
