#pragma once

#include <vector>
#include <string>
#include <iostream>
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
    Material mat(albedo, emission, BSDF_ID_ROUGHDIELECTRIC, extIor, ior, bsdfProps.getFloat("roughness", 0.1f));
    return mat;
}

inline Material make_roughconductor_material(const PropertyList& bsdfProps,
                                             const PropertyList& emitterProps = PropertyList())
{
    const Color3f emission = emitterProps.getColor("radiance",
                             emitterProps.getColor("emission", Color3f(0.f, 0.f, 0.f)));
    
    Color3f defaultEta(0.f);
    Color3f defaultK(0.f);
    
    std::string material = bsdfProps.getString("material", "none");
    for (auto& c : material) c = tolower(c);
    
    if (material == "copper" || material == "cu") {
        defaultEta = Color3f(0.200683f, 0.907677f, 1.10022f);
        defaultK   = Color3f(3.91244f, 2.45477f, 2.44215f);
    } else if (material == "gold" || material == "au") {
        defaultEta = Color3f(0.14276f, 0.38072f, 1.4552f);
        defaultK   = Color3f(3.9749f, 2.3789f, 1.598f);
    } else if (material == "aluminum" || material == "al") {
        defaultEta = Color3f(1.34138f, 0.96602f, 0.61805f);
        defaultK   = Color3f(7.33748f, 6.096f, 4.80216f);
    } else if (material == "silver" || material == "ag") {
        defaultEta = Color3f(0.05518f, 0.11728f, 1.1394f);
        defaultK   = Color3f(3.9015f, 2.45f, 2.14f);
    }
    
    const Color3f eta = bsdfProps.getColor("eta", defaultEta);
    const Color3f k   = bsdfProps.getColor("k",   defaultK);
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

inline Material make_thindielectric_material(const PropertyList& bsdfProps,
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
    Material mat(albedo, emission, BSDF_ID_THINDIELECTRIC, extIor, ior, 0.f);
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
    if (type == "thindielectric") return make_thindielectric_material(bsdfProps, emitterProps);
    if (type == "mirror")     return make_mirror_material    (bsdfProps, emitterProps);
    if (type == "diffuse")    return make_diffuse_material   (bsdfProps, emitterProps);
    if (type == "null")       return Material(Color3f(0.f), Color3f(0.f), BSDF_ID_NULL, 1.f, 1.f, 0.f);
    if (type == "microfacet") return make_microfacet_material(bsdfProps, emitterProps);
    if (type == "roughplastic") return make_roughplastic_material(bsdfProps, emitterProps);
    if (type == "plastic") {
        PropertyList props = bsdfProps;
        if (!props.hasProperty("alpha") && !props.hasProperty("roughness")) {
            props.setFloat("alpha", 0.0f);
        }
        return make_roughplastic_material(props, emitterProps);
    }
    if (type == "roughdielectric") return make_roughdielectric_material(bsdfProps, emitterProps);
    if (type == "roughconductor") return make_roughconductor_material(bsdfProps, emitterProps);
    if (type == "conductor") {
        PropertyList props = bsdfProps;
        float alpha = props.getFloat("alpha", props.getFloat("roughness", 0.0f));
        std::string material = props.getString("material", "none");
        bool hasEtaK = props.hasProperty("eta") || props.hasProperty("k");
        
        if (alpha <= 0.005f && material == "none" && !hasEtaK) {
            PropertyList mirrorProps = bsdfProps;
            Color3f spec = bsdfProps.getColor("specular_reflectance", Color3f(1.f));
            mirrorProps.setColor("reflectance", spec);
            return make_mirror_material(mirrorProps, emitterProps);
        }
        
        if (!props.hasProperty("alpha") && !props.hasProperty("roughness")) {
            props.setFloat("alpha", 0.0f);
        }
        return make_roughconductor_material(props, emitterProps);
    }

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