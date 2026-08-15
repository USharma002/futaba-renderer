#pragma once

#include "types.cuh"

FUTABA_NAMESPACE_BEGIN

enum BSDFType {
    BSDF_ID_DIFFUSE          = 0,
    BSDF_ID_DIELECTRIC       = 1,
    BSDF_ID_MIRROR           = 2,
    BSDF_ID_MICROFACET       = 3,
    BSDF_ID_NULL             = 4,
    BSDF_ID_THINDIELECTRIC   = 5,
    BSDF_ID_ROUGHDIELECTRIC  = 6,
    BSDF_ID_ROUGHCONDUCTOR   = 7,
    BSDF_ID_ROUGHPLASTIC     = 8,
};

struct Material {
    cudaTextureObject_t texObj;
    cudaTextureObject_t normalMapTexObj;

    Color3f  albedo;
    Color3f  specular;
    Color3f  emission;
    Color3f  conductorEta;
    Color3f  conductorK;

    float    alpha;
    float    extIOR;
    float    intIOR;
    BSDFType type;
    bool     isConductor;
    int      interiorMediumId = -1;
    int      exteriorMediumId = -1;

    HD Material()
        : texObj(0), normalMapTexObj(0),
          albedo(0.5f), specular(1.0f), emission(0.0f), conductorEta(0.f), conductorK(1.f),
          alpha(1.0f), extIOR(1.000277f), intIOR(1.5f), type(BSDF_ID_DIFFUSE), isConductor(false),
          interiorMediumId(-1), exteriorMediumId(-1) {}

    HD Material(const Color3f& a, const Color3f& e = Color3f(0.0f))
        : texObj(0), normalMapTexObj(0),
          albedo(a), specular(1.0f), emission(e), conductorEta(0.f), conductorK(1.f),
          alpha(1.0f), extIOR(1.000277f), intIOR(1.5f), type(BSDF_ID_DIFFUSE), isConductor(false) {}

    HD Material(const Color3f& a, const Color3f& e, BSDFType t, float extIor, float intIor, float roughness)
        : texObj(0), normalMapTexObj(0),
          albedo(a), specular(1.0f), emission(e), conductorEta(0.f), conductorK(1.f),
          alpha(roughness), extIOR(extIor), intIOR(intIor), type(t), isConductor(false) {}

    /// Returns true if this material type should not block shadow rays
    /// (i.e., it is fully transmissive for direct-lighting visibility tests).
    HD static bool isShadowTransparent(BSDFType t) {
        return t == BSDF_ID_NULL || t == BSDF_ID_THINDIELECTRIC;
    }
};

FUTABA_NAMESPACE_END
