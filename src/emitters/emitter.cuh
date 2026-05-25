#pragma once

#include "types.cuh"
#include "emitter_sample.cuh"

namespace futaba {

// ---------------------------------------------------------------------------
// Emitter type constants (GPU-visible).
// Values MUST match EmitterType enum in parser/scene_loader.h.
// ---------------------------------------------------------------------------
static constexpr uint32_t kEmitterTypeNone        = 0u;
static constexpr uint32_t kEmitterTypeArea        = 1u;
static constexpr uint32_t kEmitterTypePoint       = 2u;
static constexpr uint32_t kEmitterTypeDirectional = 3u;

enum EmitterFlags : uint32_t {
    EMITTER_FLAG_TWO_SIDED = 1u << 0,
};

struct EmitterGPU {
    uint32_t  type;           // One of kEmitterType* constants above
    uint32_t  flags;          // Bitfield - see EmitterFlags
    Color3f   radiance;
    Point3f   position;       // Point emitters
    Vector3f  direction;      // Directional emitters
    int       attachedMeshId; // -1 unless area emitter; CPU also tracks this
};

} // namespace futaba
