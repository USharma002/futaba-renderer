#pragma once

#include "types.cuh"
#include "bbox.cuh"
#include "ppg/stree.h"

FUTABA_NAMESPACE_BEGIN

enum class GuidingType : uint32_t {
    None = 0,
    PPG  = 1,
};

// Everything the GPU integrator needs from path guiding, in one flat POD.
struct GuidingParams {
    bool        active      = false;            // guided sampling enabled
    GuidingType type        = GuidingType::None;// active guiding model
    STreeNode*  sTreeNodes  = nullptr;          // device S-tree nodes (built on CPU)
    AABB        sTreeAABB;
    float       bsdf_sampling_fraction = 0.5f;

    float*    train_active    = nullptr;
    Point3f*  train_position  = nullptr;
    Vector3f* train_wo        = nullptr;
    Color3f*  train_radiance  = nullptr;
    float*    train_pdf       = nullptr;
    int       train_max_depth = 0;
};

FUTABA_NAMESPACE_END
