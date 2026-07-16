#pragma once

#include <cstdint>
#include "types.cuh"
#include "bbox.cuh"
#include "emitter.cuh"

FUTABA_NAMESPACE_BEGIN

struct MeshInstance {
    uint32_t     triangleStart = 0;   // First triangle index in the flat array
    uint32_t     triangleCount = 0;
    Matrix4f     transform;           // World transform (already baked into triangles)
    AABB         bbox;                // World-space bounding box
    int          materialId = -1;     // Index into LoadedScene::materials
    EmitterType  emitterType = EmitterType::None;
    int          emitterId   = -1;    // Index into LoadedScene::emitters (-1 = none)
};

FUTABA_NAMESPACE_END