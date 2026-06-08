#pragma once

#include "scene_loader.h"
#include "scene.cuh"
#include "texture_manager.h"
#include "medium.cuh"

namespace futaba {

struct UploadedSceneConfig {
    bool hasCamera = false;
    Point3f camOrigin;
    Point3f camTarget;
    ::Vector3f camUp;
    float camFov = 45.f;
    float currentFocusDistance = 1.f;
    std::string integratorType = "path";
    uint32_t triangleCount = 0;
    bool hasMedium = false;
    int mediumMeshId = -1;
    Medium medium;
};

class SceneUploader {
public:
    static UploadedSceneConfig upload(
        const LoadedScene& loaded,
        const std::string& xmlPath,
        Scene& scene,
        TextureManager& textureManager,
        bool useVertexNormals,
        bool useNEE
    );
};

} // namespace futaba
