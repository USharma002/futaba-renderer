#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include "proplist.h"
#include "scene_loader.h"

namespace futaba {

EmitterInstance make_area_emitter(const PropertyList& props);
EmitterInstance make_point_emitter(const PropertyList& props);
EmitterInstance make_directional_emitter(const PropertyList& props);

using EmitterBuilderFn = EmitterInstance (*)(const PropertyList&);

const std::unordered_map<std::string, EmitterBuilderFn>& emitterBuilderRegistry();

EmitterInstance makeEmitterFromPropertyLists(
        const std::string&        type,
        const PropertyList&       props,
        std::vector<std::string>& warnings);

} // namespace futaba