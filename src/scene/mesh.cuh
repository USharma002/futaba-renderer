#pragma once

#include <cstdint>

namespace futaba {

// GPU-side mesh instance metadata (lightweight; fit for device transfer).
struct MeshInstanceGPU {
    uint32_t triangleStart;
    uint32_t triangleCount;
    int      emitterId; // index into Scene.emitters (-1 = not emissive)
};

} // namespace futaba
