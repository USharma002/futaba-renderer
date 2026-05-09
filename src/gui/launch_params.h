#pragma once
#include <vector_types.h>
#include "types.cuh"
#include "perspective.cuh"
#include "scene.cuh"

namespace futaba {

enum IntegratorMode {
    INTEGRATOR_PATH = 0,
    INTEGRATOR_NORMALS = 1,
    INTEGRATOR_HEATMAP = 2
};

struct LaunchParams {
    uchar4* pbo_ptr;
    Color3f* film_pixels;
    int width;
    int height;
    int sampleCount;
    PerspectiveCamera camera;
    Scene scene;
    int max_depth;
    int rr_depth;
    int integrator_mode;
    bool use_antialiasing;
};

} // namespace futaba
