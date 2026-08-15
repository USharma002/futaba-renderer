#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include "types.cuh"

#include "material.cuh"
#include "triangle.cuh"
#include "bbox.cuh"
#include "emitter.cuh"
#include "mesh.cuh"
#include "medium.cuh"

FUTABA_NAMESPACE_BEGIN

// Camera staging configuration for GPU upload.
struct CameraSettings {
    bool       hasCamera = false;
    Point3f    origin = Point3f(0.f);
    Point3f    target = Point3f(0.f);
    Vector3f   up  = Vector3f(0.f, 1.f, 0.f);
    float      fov = 45.f;
};

// Environment map staging configuration.
struct EnvMapSettings {
    bool                 hasEnvMap = false;
    std::vector<Color3f> pixels; // CPU staging buffer (will be uploaded as CUDA texture)
    int                  width = 0;
    int                  height = 0;
    Matrix4f             toWorld;

    bool                 hasConstant = false;
    Color3f              constantRadiance = Color3f(0.f);
};

// Result of a successful scene load (staging buffers for GPU allocation).
struct CPUScene {
    std::vector<Triangle>        triangles;
    std::vector<Material>        materials;
    std::vector<std::string>     materialTexturePaths; // Same size as CPUScene::materials
    std::vector<std::string>     materialNormalMapPaths; // Same size as CPUScene::materials
    std::vector<EmitterInstance>  emitters;
    std::vector<MeshInstance>    meshes;
    std::vector<std::string>     meshNames;            // CPU-only names, parallel to meshes
    std::vector<Medium>          media;
    std::vector<std::string>     mediumNames;

    CameraSettings  camera;
    EnvMapSettings  envMap;

    std::string integratorType = "path";

    // Non-fatal warnings collected during loading (e.g. unknown BSDF/emitter types).
    // Check this after a successful load to detect degraded scenes.
    std::vector<std::string> warnings;
};

// ---------------------------------------------------------------------------
// SceneLoader
// ---------------------------------------------------------------------------
class SceneLoader {
public:
    bool load(const std::string& xmlPath,
              CPUScene&          out,
              std::string&       errorOut);
};

FUTABA_NAMESPACE_END