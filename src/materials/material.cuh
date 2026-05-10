#pragma once

#include "types.cuh"

namespace futaba {

enum BSDFType {
    BSDF_ID_DIFFUSE          = 0,
    BSDF_ID_DIELECTRIC       = 1,
    BSDF_ID_MIRROR           = 2,
    BSDF_ID_MICROFACET       = 3,
    BSDF_ID_ROUGHCONDUCTOR   = 4,
    BSDF_ID_ROUGHPLASTIC     = 5,
    BSDF_ID_ROUGHDIELECTRIC  = 6,
};

struct Material {
    Color3f  albedo;
    Color3f  specular;
    Color3f  emission;
    Color3f  conductorEta;
    Color3f  conductorK;
    float    alpha;
    float    extIOR;
    float    intIOR;
    bool     isConductor;
    BSDFType type;

    HD Material()
        : albedo(0.5f), specular(1.0f), emission(0.0f), conductorEta(0.f), conductorK(1.f),
          alpha(1.0f), extIOR(1.000277f), intIOR(1.5f), isConductor(false), type(BSDF_ID_DIFFUSE) {}

    HD Material(const Color3f& a, const Color3f& e = Color3f(0.0f))
        : albedo(a), specular(1.0f), emission(e), conductorEta(0.f), conductorK(1.f),
          alpha(1.0f), extIOR(1.000277f), intIOR(1.5f), isConductor(false), type(BSDF_ID_DIFFUSE) {}

    HD Material(const Color3f& a, const Color3f& e, BSDFType t, float extIor, float intIor, float roughness)
        : albedo(a), specular(1.0f), emission(e), conductorEta(0.f), conductorK(1.f),
          alpha(roughness), extIOR(extIor), intIOR(intIor), isConductor(false), type(t){}
};

} // namespace futaba
