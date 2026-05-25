#pragma once
#include <vector_types.h>
#include "types.cuh"
#include "perspective.cuh"
#include "scene.cuh"
#include "guiding.h"

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
    TONEMAPPING_REINHARD = 2,
    TONEMAPPING_FILMIC = 3
};

struct LaunchParams {
    uchar4* pbo_ptr;
    Color3f* film_pixels;
    Scene scene;
    PerspectiveCamera camera;
    int width;
    int height;
    int sampleCount;
    int max_depth;
    int rr_depth;
    int integrator_mode;
    int tonemapping_mode;
    Vector3f phong_light_dir;
    float phong_ambient;
    float phong_diffuse;
    float phong_specular;
    float phong_shininess;
    bool use_antialiasing;
    bool denoise_active;
    Color3f* denoise_albedo_buffer;
    Color3f* denoise_normal_buffer;
    int path_guiding_mode;

    // Training buffers (shape: max_depth x height x width)
    float* train_active;
    Point3f* train_position;
    Color3f* train_normals;
    Color3f* train_wi;
    Color3f* train_wo;
    Color3f* train_radiance;
    float* train_material_id;

    // Visualization parameters
    uchar4* vis_pbo_ptr;
    int vis_depth;
    int vis_buffer_type;
};

} // namespace futaba
