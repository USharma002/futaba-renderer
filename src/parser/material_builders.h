#pragma once

#include <vector>
#include <string>
#include "proplist.h"
#include "material.cuh"

namespace futaba {

inline Material make_diffuse_material(const PropertyList& bsdfProps,
                                      const PropertyList& emitterProps = PropertyList())
{
    const Color3f albedo   = bsdfProps.getColor("albedo",
                             bsdfProps.getColor("reflectance", Color3f(0.8f, 0.8f, 0.8f)));
    const Color3f emission = emitterProps.getColor("radiance",
                             emitterProps.getColor("emission", Color3f(0.f, 0.f, 0.f)));
    return Material(albedo, emission, BSDF_ID_DIFFUSE, 1.5f);
}

inline Material make_dielectric_material(const PropertyList& bsdfProps,
                                         const PropertyList& emitterProps = PropertyList())
{
    const Color3f albedo   = bsdfProps.getColor("albedo",
                             bsdfProps.getColor("reflectance", Color3f(1.f, 1.f, 1.f)));
    const Color3f emission = emitterProps.getColor("radiance",
                             emitterProps.getColor("emission", Color3f(0.f, 0.f, 0.f)));
    const float   ior      = bsdfProps.getFloat("intIOR",
                             bsdfProps.getFloat("ior", 1.5f));
    return Material(albedo, emission, BSDF_ID_DIELECTRIC, ior);
}

inline Material make_mirror_material(const PropertyList& bsdfProps,
                                     const PropertyList& emitterProps = PropertyList())
{
    const Color3f albedo   = bsdfProps.getColor("albedo",
                             bsdfProps.getColor("reflectance", Color3f(1.f, 1.f, 1.f)));
    const Color3f emission = emitterProps.getColor("radiance",
                             emitterProps.getColor("emission", Color3f(0.f, 0.f, 0.f)));
    return Material(albedo, emission, BSDF_ID_MIRROR, 1.f);
}

// Build a Material from parsed BSDF and emitter property lists.
// Unknown BSDF types fall back to diffuse and append a diagnostic to `warnings`.
inline Material makeMaterialFromPropertyLists(
        const PropertyList&       bsdfProps,
        const PropertyList&       emitterProps,
        std::vector<std::string>& warnings)
{
    const std::string type = bsdfProps.getString("type", "diffuse");

    if (type == "dielectric") return make_dielectric_material(bsdfProps, emitterProps);
    if (type == "mirror")     return make_mirror_material    (bsdfProps, emitterProps);
    if (type == "diffuse")    return make_diffuse_material   (bsdfProps, emitterProps);

    // Unknown type: fall back to diffuse and record a warning.
    warnings.push_back(
        "Unknown BSDF type '" + type + "' – falling back to diffuse. "
        "Check the spelling or add a builder to material_builders.h.");
    return make_diffuse_material(bsdfProps, emitterProps);
}

// Legacy overload for call sites that don't have a warnings vector yet.
// Prefer the three-argument version where possible.
inline Material makeMaterialFromPropertyLists(const PropertyList& bsdfProps,
                                              const PropertyList& emitterProps)
{
    std::vector<std::string> sink;
    return makeMaterialFromPropertyLists(bsdfProps, emitterProps, sink);
}

} // namespace futaba