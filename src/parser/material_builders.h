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
    Material mat(albedo, emission, BSDF_ID_DIFFUSE, 1.000277f, 1.5f, 1.f);
    return mat;
}

inline Material make_dielectric_material(const PropertyList& bsdfProps,
                                         const PropertyList& emitterProps = PropertyList())
{
    const Color3f albedo   = bsdfProps.getColor("albedo",
                             bsdfProps.getColor("reflectance", Color3f(1.f, 1.f, 1.f)));
    const Color3f emission = emitterProps.getColor("radiance",
                             emitterProps.getColor("emission", Color3f(0.f, 0.f, 0.f)));
    const float   extIor   = bsdfProps.getFloat("extIOR",
                             bsdfProps.getFloat("ext_ior", 1.000277f));
    const float   ior      = bsdfProps.getFloat("intIOR",
                             bsdfProps.getFloat("int_ior",
                             bsdfProps.getFloat("ior", 1.5f)));
    Material mat(albedo, emission, BSDF_ID_DIELECTRIC, extIor, ior, 1.f);
    return mat;
}

inline Material make_mirror_material(const PropertyList& bsdfProps,
                                     const PropertyList& emitterProps = PropertyList())
{
    const Color3f albedo   = bsdfProps.getColor("albedo",
                             bsdfProps.getColor("reflectance", Color3f(1.f, 1.f, 1.f)));
    const Color3f emission = emitterProps.getColor("radiance",
                             emitterProps.getColor("emission", Color3f(0.f, 0.f, 0.f)));
    Material mat(albedo, emission, BSDF_ID_MIRROR, 1.000277f, 1.f, 1.f);
    return mat;
}


inline Material make_microfacet_material(const PropertyList& bsdfProps,
                                     const PropertyList& emitterProps = PropertyList())
{
    const Color3f albedo   = bsdfProps.getColor("kd",
                             bsdfProps.getColor("reflectance", Color3f(1.f, 1.f, 1.f)));
    const Color3f emission = emitterProps.getColor("radiance",
                             emitterProps.getColor("emission", Color3f(0.f, 0.f, 0.f)));
    const float extIor = bsdfProps.getFloat("extIOR",
                         bsdfProps.getFloat("ext_ior", 1.000277f));
    const float intIor = bsdfProps.getFloat("intIOR",
                         bsdfProps.getFloat("int_ior",
                         bsdfProps.getFloat("ior", 1.5046f)));
    const float alpha  = bsdfProps.getFloat("alpha",
                         bsdfProps.getFloat("roughness", 0.1f));
    Material mat(albedo, emission, BSDF_ID_MICROFACET, extIor, intIor, alpha);
    mat.isConductor = false;
    return mat;
}

inline Material make_roughplastic_material(const PropertyList& bsdfProps,
                                           const PropertyList& emitterProps = PropertyList())
{
    const Color3f albedo   = bsdfProps.getColor("diffuse_reflectance",
                             bsdfProps.getColor("reflectance", Color3f(0.5f, 0.5f, 0.5f)));
    const Color3f emission = emitterProps.getColor("radiance",
                             emitterProps.getColor("emission", Color3f(0.f, 0.f, 0.f)));
    const float extIor = bsdfProps.getFloat("extIOR",
                         bsdfProps.getFloat("ext_ior", 1.000277f));
    const float intIor = bsdfProps.getFloat("intIOR",
                         bsdfProps.getFloat("int_ior",
                         bsdfProps.getFloat("ior", 1.5046f)));
    const float alpha  = bsdfProps.getFloat("alpha",
                         bsdfProps.getFloat("roughness", 0.1f));
    Material mat(albedo, emission, BSDF_ID_MICROFACET, extIor, intIor, alpha);
    mat.isConductor = false;
    return mat;
}

inline Material make_roughdielectric_material(const PropertyList& bsdfProps,
                                             const PropertyList& emitterProps = PropertyList())
{
    const Color3f albedo   = bsdfProps.getColor("albedo",
                             bsdfProps.getColor("reflectance", Color3f(1.f, 1.f, 1.f)));
    const Color3f emission = emitterProps.getColor("radiance",
                             emitterProps.getColor("emission", Color3f(0.f, 0.f, 0.f)));
    const float extIor = bsdfProps.getFloat("extIOR",
                         bsdfProps.getFloat("ext_ior", 1.000277f));
    const float ior    = bsdfProps.getFloat("intIOR",
                         bsdfProps.getFloat("int_ior",
                         bsdfProps.getFloat("ior", 1.5f)));
    Material mat(albedo, emission, BSDF_ID_DIELECTRIC, extIor, ior, bsdfProps.getFloat("roughness", 0.1f));
    return mat;
}

inline Material make_roughconductor_material(const PropertyList& bsdfProps,
                                             const PropertyList& emitterProps = PropertyList())
{
    const Color3f emission = emitterProps.getColor("radiance",
                             emitterProps.getColor("emission", Color3f(0.f, 0.f, 0.f)));
    const Color3f eta = bsdfProps.getColor("eta", Color3f(0.2f, 0.9f, 1.1f));
    const Color3f k   = bsdfProps.getColor("k",   Color3f(3.9f, 2.5f, 2.4f));
    const Color3f spec = bsdfProps.getColor("specular_reflectance", Color3f(1.f));
    const float extIor = bsdfProps.getFloat("extIOR",
                         bsdfProps.getFloat("ext_ior", 1.000277f));
    const float alpha  = bsdfProps.getFloat("alpha",
                         bsdfProps.getFloat("roughness", 0.1f));

    Material mat(Color3f(0.f), emission, BSDF_ID_MICROFACET, extIor, 1.f, alpha);
    mat.isConductor = true;
    mat.specular = spec;
    mat.conductorEta = eta;
    mat.conductorK = k;
    return mat;
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
    if (type == "microfacet") return make_microfacet_material(bsdfProps, emitterProps);
    if (type == "roughplastic") return make_roughplastic_material(bsdfProps, emitterProps);
    if (type == "roughdielectric") return make_roughdielectric_material(bsdfProps, emitterProps);
    if (type == "roughconductor") return make_roughconductor_material(bsdfProps, emitterProps);

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