#pragma once

#include <string>
#include <vector>
#include "types.cuh"
#include "perspective.cuh"

namespace futaba {
    struct Triangle;
    struct Material;
}

namespace futaba {

struct LoadedScene {
    std::vector<Triangle>  triangles;
    std::vector<Material>  materials;

    bool       hasCamera = false;
    ::Point3f  camOrigin;
    ::Point3f  camTarget;
    ::Vector3f camUp    = ::Vector3f(0.f, 1.f, 0.f);
    float      camFov   = 45.f;
};

class SceneLoader {
public:
    bool load(const std::string& xmlPath, LoadedScene& out, std::string& errorOut);

private:
    bool parseMesh(const std::string& baseDir,
                   const std::string& objFilename,
                   int                materialId,
                   const Matrix4f&    transform,
                   LoadedScene&       out,
                   std::string&       errorOut);

    // xmlPath removed — it was unused.
    bool parseCamera(const std::string& originStr,
                     const std::string& targetStr,
                     const std::string& upStr,
                     float              fov,
                     LoadedScene&       out,
                     std::string&       errorOut);

    static ::Vector3f parseVec3(const std::string& s);
};

} // namespace futaba