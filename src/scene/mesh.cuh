#pragma once

#include <cstdint>
#include "types.cuh"
#include "bbox.cuh"
#include "emitter.cuh"

FUTABA_NAMESPACE_BEGIN

struct MeshInstance {
    Matrix4f     transform;                        // World transform (64 bytes)
    AABB         bbox;                             // World-space bounding box (24 bytes)
    uint32_t     triangleStart = 0;                // First triangle index (4 bytes)
    uint32_t     triangleCount = 0;                // Number of triangles (4 bytes)
    int          materialId    = -1;               // Material index (4 bytes)
    int          emitterId     = -1;               // Emitter index (4 bytes)
    EmitterType  emitterType   = EmitterType::None;// Emitter type enum (4 bytes)
};

FUTABA_NAMESPACE_END