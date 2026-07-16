#include "emitter_builders.h"

FUTABA_NAMESPACE_BEGIN

EmitterInstance make_area_emitter(const PropertyList& props) {
    EmitterInstance inst(EmitterType::Area);
    inst.radiance = props.getColor("radiance",
                    props.getColor("emission", Color3f(0.f, 0.f, 0.f)));
    inst.twoSided = props.getBoolean("two_sided", true);
    return inst;
}

EmitterInstance make_point_emitter(const PropertyList& props) {
    EmitterInstance inst(EmitterType::Point);
    inst.radiance = props.getColor("radiance", Color3f(0.f));
    inst.position = props.getPoint("position",  Point3f(0.f));
    return inst;
}

EmitterInstance make_directional_emitter(const PropertyList& props) {
    EmitterInstance inst(EmitterType::Directional);
    inst.radiance  = props.getColor("radiance",
                     props.getColor("irradiance", Color3f(0.f)));
    inst.direction = normalize(props.getVector("direction", Vector3f(0.f, -1.f, 0.f)));
    return inst;
}

const std::unordered_map<std::string, EmitterBuilderFn>& emitterBuilderRegistry() {
    static const std::unordered_map<std::string, EmitterBuilderFn> kBuilders = {
        { "area",            make_area_emitter        },
        { "point",           make_point_emitter       },
        { "directional",     make_directional_emitter },
        // Partial aliases – routed to the closest available implementation.
        // TODO: implement constant (env) and directionalarea properly.
        { "constant",        make_directional_emitter },
        { "directionalarea", make_directional_emitter },
        { "envmap",          make_directional_emitter },
    };
    return kBuilders;
}

EmitterInstance makeEmitterFromPropertyLists(
        const std::string&        type,
        const PropertyList&       props,
        std::vector<std::string>& warnings)
{
    const std::string key = type.empty() ? std::string("area") : type;
    const auto& registry  = emitterBuilderRegistry();
    const auto  it        = registry.find(key);

    if (it != registry.end())
        return it->second(props);

    // Unknown type: produce a no-op emitter and warn.
    warnings.push_back(
        "Unknown emitter type '" + key + "' - treated as no emitter. "
        "Check the spelling or add a builder to emitter_builders.h.");
    return EmitterInstance(EmitterType::None);
}

FUTABA_NAMESPACE_END
