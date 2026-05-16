#pragma once
#include <vector_types.h>
#include "types.cuh"
#include "perspective.cuh"
#include "scene.cuh"

namespace futaba {

enum IntegratorMode {
    INTEGRATOR_PATH = 0,
    INTEGRATOR_NORMALS = 1,
    INTEGRATOR_DEPTH = 2,
    INTEGRATOR_ALBEDO = 3,
    INTEGRATOR_PHONG = 4,
    INTEGRATOR_PRIMITIVES = 5,
    INTEGRATOR_HEATMAP = 6,
    INTEGRATOR_VOLPATH = 7
};

enum TonemappingMode {
    TONEMAPPING_NONE = 0,
    TONEMAPPING_ACES = 1,
    TONEMAPPING_REINHARDT = 2,
    TONEMAPPING_FILMIC = 3
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
    int tonemapping_mode;
    bool use_antialiasing;
    Vector3f phong_light_dir;
    float phong_ambient;
    float phong_diffuse;
    float phong_specular;
    float phong_shininess;
};

} // namespace futaba
