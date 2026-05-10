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

// ---------------------------------------------------------------------------
// Emitter type enum (CPU side).
// GPU-side constants in scene/scene.cuh (kEmitterType*) must stay in sync.
// ---------------------------------------------------------------------------
enum class EmitterType : uint32_t {
    None        = 0,
    Area        = 1,
    Point       = 2,
    Directional = 3,
};

// Generic emitter instance produced by the parser.
struct EmitterInstance {
    EmitterType type      = EmitterType::None;
    Color3f     radiance  = Color3f(0.f);
    Point3f     position  = Point3f(0.f, 0.f, 0.f); // point emitters
    ::Vector3f  direction = ::Vector3f(0.f, -1.f, 0.f); // directional emitters
    bool        twoSided  = true; // area emitters: emit from both faces unless false

    EmitterInstance() = default;
    explicit EmitterInstance(EmitterType t) : type(t) {}
};

// Per-mesh metadata kept alongside the flat triangle array.
struct MeshInstance {
    std::string  name;
    uint32_t     triangleStart;   // First triangle index in the flat array
    uint32_t     triangleCount;
    Matrix4f     transform;       // World transform (already baked into triangles)
    Point3f      boundingBoxMin;  // World-space AABB corners
    Point3f      boundingBoxMax;
    int          materialId;      // Index into LoadedScene::materials
    EmitterType  emitterType = EmitterType::None;
    int          emitterId   = -1; // Index into LoadedScene::emitters (-1 = none)
};

// Result of a successful scene load.
struct LoadedScene {
    std::vector<Triangle>       triangles;
    std::vector<Material>       materials;
    std::vector<EmitterInstance> emitters;
    std::vector<MeshInstance>   meshes;

    bool       hasCamera = false;
    Point3f    camOrigin;
    Point3f    camTarget;
    ::Vector3f camUp  = ::Vector3f(0.f, 1.f, 0.f);
    float      camFov = 45.f;

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
              LoadedScene&       out,
              std::string&       errorOut);

private:
    // Parses one OBJ mesh file and appends triangles / mesh instance to out.
    // normalTransform is the inverse-transpose of the upper-left 3×3 of
    // transform, used to correctly transform vertex normals under non-uniform scale.
    bool parseMesh(const std::string& baseDir,
                   const std::string& objFilename,
                   const std::string& meshName,
                   int                materialId,
                   int                emitterId,      // -1 if not emissive
                   const Matrix4f&    transform,
                   const Matrix4f&    normalTransform,
                   LoadedScene&       out,
                   std::string&       errorOut);

    bool parseCamera(const std::string& originStr,
                     const std::string& targetStr,
                     const std::string& upStr,
                     float              fov,
                     LoadedScene&       out,
                     std::string&       errorOut);

    static Vector3f parseVec3(const std::string& s);
};

} // namespace futaba