#include "material_builders.h"
#include <cctype>
#include <algorithm>

FUTABA_NAMESPACE_BEGIN

namespace {
    inline Color3f get_albedo_prop(const PropertyList& props, const Color3f& def = Color3f(1.f)) {
        return props.getColor("albedo", props.getColor("reflectance", props.getColor("kd", props.getColor("diffuse_reflectance", def))));
    }

    inline Color3f get_emission_prop(const PropertyList& props) {
        return props.getColor("radiance", props.getColor("emission", Color3f(0.f)));
    }

    inline float get_roughness_prop(const PropertyList& props, float def = 0.1f) {
        return props.getFloat("alpha", props.getFloat("roughness", def));
    }

    inline float get_ext_ior_prop(const PropertyList& props, float def = 1.000277f) {
        return props.getFloat("extIOR", props.getFloat("ext_ior", def));
    }

    inline float get_int_ior_prop(const PropertyList& props, float def = 1.5f) {
        return props.getFloat("intIOR", props.getFloat("int_ior", props.getFloat("ior", def)));
    }
}

Material make_diffuse_material(const PropertyList& bsdfProps, const PropertyList& emitterProps) {
    return Material(get_albedo_prop(bsdfProps, Color3f(0.8f)), get_emission_prop(emitterProps),
                    BSDF_ID_DIFFUSE, 1.000277f, 1.5f, 1.f);
}

Material make_dielectric_material(const PropertyList& bsdfProps, const PropertyList& emitterProps) {
    return Material(get_albedo_prop(bsdfProps, Color3f(1.f)), get_emission_prop(emitterProps),
                    BSDF_ID_DIELECTRIC, get_ext_ior_prop(bsdfProps, 1.000277f), get_int_ior_prop(bsdfProps, 1.5f), 0.f);
}

Material make_mirror_material(const PropertyList& bsdfProps, const PropertyList& emitterProps) {
    return Material(get_albedo_prop(bsdfProps, Color3f(1.f)), get_emission_prop(emitterProps),
                    BSDF_ID_MIRROR, 1.000277f, 1.f, 0.f);
}

Material make_microfacet_material(const PropertyList& bsdfProps, const PropertyList& emitterProps) {
    Material mat(get_albedo_prop(bsdfProps, Color3f(1.f)), get_emission_prop(emitterProps),
                 BSDF_ID_MICROFACET, get_ext_ior_prop(bsdfProps, 1.000277f),
                 get_int_ior_prop(bsdfProps, 1.5046f), get_roughness_prop(bsdfProps, 0.1f));
    mat.isConductor = false;
    return mat;
}

Material make_roughplastic_material(const PropertyList& bsdfProps, const PropertyList& emitterProps) {
    Material mat(get_albedo_prop(bsdfProps, Color3f(0.5f)), get_emission_prop(emitterProps),
                 BSDF_ID_ROUGHPLASTIC, get_ext_ior_prop(bsdfProps, 1.000277f),
                 get_int_ior_prop(bsdfProps, 1.5046f), get_roughness_prop(bsdfProps, 0.1f));
    mat.isConductor = false;
    return mat;
}

Material make_roughdielectric_material(const PropertyList& bsdfProps, const PropertyList& emitterProps) {
    return Material(get_albedo_prop(bsdfProps, Color3f(1.f)), get_emission_prop(emitterProps),
                    BSDF_ID_ROUGHDIELECTRIC, get_ext_ior_prop(bsdfProps, 1.000277f),
                    get_int_ior_prop(bsdfProps, 1.5f), get_roughness_prop(bsdfProps, 0.1f));
}

Material make_roughconductor_material(const PropertyList& bsdfProps, const PropertyList& emitterProps) {
    Color3f defaultEta(0.f), defaultK(0.f);
    std::string material = bsdfProps.getString("material", "none");
    for (auto& c : material) c = (char)std::tolower((unsigned char)c);

    struct ConductorConsts { const char* name; Color3f eta, k; };
    static const ConductorConsts kConductors[] = {
        { "copper",   {0.200683f, 0.907677f, 1.10022f}, {3.91244f, 2.45477f, 2.44215f} },
        { "cu",       {0.200683f, 0.907677f, 1.10022f}, {3.91244f, 2.45477f, 2.44215f} },
        { "gold",     {0.14276f, 0.38072f, 1.4552f},    {3.9749f, 2.3789f, 1.598f} },
        { "au",       {0.14276f, 0.38072f, 1.4552f},    {3.9749f, 2.3789f, 1.598f} },
        { "aluminum", {1.34138f, 0.96602f, 0.61805f},   {7.33748f, 6.096f, 4.80216f} },
        { "al",       {1.34138f, 0.96602f, 0.61805f},   {7.33748f, 6.096f, 4.80216f} },
        { "silver",   {0.048f, 0.050f, 0.070f},         {4.10f,  3.60f,  2.65f} },
        { "ag",       {0.048f, 0.050f, 0.070f},         {4.10f,  3.60f,  2.65f} }
    };

    for (const auto& c : kConductors) {
        if (material == c.name) {
            defaultEta = c.eta;
            defaultK   = c.k;
            break;
        }
    }

    Material mat(Color3f(0.f), get_emission_prop(emitterProps), BSDF_ID_ROUGHCONDUCTOR,
                 get_ext_ior_prop(bsdfProps, 1.000277f), 1.f, get_roughness_prop(bsdfProps, 0.1f));
    mat.isConductor   = true;
    mat.specular      = bsdfProps.getColor("specular_reflectance", Color3f(1.f));
    mat.conductorEta  = bsdfProps.getColor("eta", defaultEta);
    mat.conductorK    = bsdfProps.getColor("k",   defaultK);
    return mat;
}

Material make_thindielectric_material(const PropertyList& bsdfProps, const PropertyList& emitterProps) {
    return Material(get_albedo_prop(bsdfProps, Color3f(1.f)), get_emission_prop(emitterProps),
                    BSDF_ID_THINDIELECTRIC, get_ext_ior_prop(bsdfProps, 1.000277f),
                    get_int_ior_prop(bsdfProps, 1.5f), 0.f);
}


Material makeMaterialFromPropertyLists(
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

Material makeMaterialFromPropertyLists(const PropertyList& bsdfProps,
                                      const PropertyList& emitterProps)
{
    std::vector<std::string> sink;
    return makeMaterialFromPropertyLists(bsdfProps, emitterProps, sink);
}

FUTABA_NAMESPACE_END
