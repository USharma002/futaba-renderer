#pragma once

#include "proplist.h"
#include "material.cuh"

namespace futaba {

inline Material make_diffuse_material(const PropertyList& bsdfProps,
                                      const PropertyList& emitterProps = PropertyList())
{
    Color3f albedo   = bsdfProps.getColor("albedo",
                       bsdfProps.getColor("reflectance", ::Color3f(0.8f, 0.8f, 0.8f)));
    Color3f emission = emitterProps.getColor("radiance",
                       emitterProps.getColor("emission", ::Color3f(0.f, 0.f, 0.f)));
    return Material(albedo, emission, BSDF_ID_DIFFUSE, 1.5f);
}

inline Material make_dielectric_material(const PropertyList& bsdfProps,
                                         const PropertyList& emitterProps = PropertyList())
{
    Color3f albedo   = bsdfProps.getColor("albedo",
                       bsdfProps.getColor("reflectance", ::Color3f(1.f, 1.f, 1.f)));
    Color3f emission = emitterProps.getColor("radiance",
                       emitterProps.getColor("emission", ::Color3f(0.f, 0.f, 0.f)));
    float   ior      = bsdfProps.getFloat("intIOR", bsdfProps.getFloat("ior", 1.5f));
    return Material(albedo, emission, BSDF_ID_DIELECTRIC, ior);
}

inline Material make_mirror_material(const PropertyList& bsdfProps,
                                     const PropertyList& emitterProps = PropertyList())
{
    Color3f albedo   = bsdfProps.getColor("albedo",
                       bsdfProps.getColor("reflectance", ::Color3f(1.f, 1.f, 1.f)));
    Color3f emission = emitterProps.getColor("radiance",
                       emitterProps.getColor("emission", ::Color3f(0.f, 0.f, 0.f)));
    return Material(albedo, emission, BSDF_ID_MIRROR, 1.f);
}

inline Material makeMaterialFromPropertyLists(const PropertyList& bsdfProps,
                                              const PropertyList& emitterProps)
{
    const std::string type = bsdfProps.getString("type", "diffuse");
    if (type == "dielectric") return make_dielectric_material(bsdfProps, emitterProps);
    if (type == "mirror")     return make_mirror_material    (bsdfProps, emitterProps);
    return                           make_diffuse_material   (bsdfProps, emitterProps);
}

} // namespace futaba