#pragma once
#include <vector_types.h>
#include "types.cuh"
#include "perspective.cuh"
#include "scene.cuh"
#include "emitter_sample.cuh"

FUTABA_NAMESPACE_BEGIN

enum IntegratorMode {
    INTEGRATOR_PATH = 0,
    INTEGRATOR_NORMALS = 1,
    INTEGRATOR_DEPTH = 2,
    INTEGRATOR_ALBEDO = 3,
    INTEGRATOR_PHONG = 4,
    INTEGRATOR_PRIMITIVES = 5,
    INTEGRATOR_HEATMAP = 6
};

enum LightSamplerType {
    LIGHT_SAMPLER_POWER = 0
};

enum TonemappingMode {
    TONEMAPPING_NONE = 0,
    TONEMAPPING_ACES = 1,
    TONEMAPPING_REINHARD = 2,
    TONEMAPPING_FILMIC = 3
};

// Debug-integrator params. Only read by the Phong integrator.
struct PhongParams {
    Vector3f light_dir;
    float    ambient;
    float    diffuse;
    float    specular;
    float    shininess;
};

// Denoiser guide buffers: filled by PathRecorder during the path integrator's
// first bounce, consumed by the OptiX denoiser after the frame is rendered.
struct DenoiseParams {
    bool     active = false;
    Color3f* albedo_buffer = nullptr;
    Color3f* normal_buffer = nullptr;
};

struct LightSamplerData {
    int type;
    CDFLightSamplerData cdf;
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
    LightSamplerData light_sampler;
    bool use_antialiasing;
    PhongParams phong;
    DenoiseParams denoise;
};

FUTABA_NAMESPACE_END