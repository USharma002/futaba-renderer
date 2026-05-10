#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include "triangle.cuh"
#include "material.cuh"
#include "bvh.cuh"

namespace futaba {

// ---------------------------------------------------------------------------
// CUDA error-checking helper (host only).
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t _err = (call);                                       \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_err));             \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Emitter type constants (GPU-visible).
// Values MUST match EmitterType enum in parser/scene_loader.h.
// ---------------------------------------------------------------------------
static constexpr uint32_t kEmitterTypeNone        = 0u;
static constexpr uint32_t kEmitterTypeArea        = 1u;
static constexpr uint32_t kEmitterTypePoint       = 2u;
static constexpr uint32_t kEmitterTypeDirectional = 3u;

// GPU-side mesh instance metadata (lightweight; fit for device transfer).
struct MeshInstanceGPU {
    uint32_t triangleStart;
    uint32_t triangleCount;
    int      emitterId; // index into Scene.emitters (-1 = not emissive)
};

struct EmitterGPU {
    uint32_t  type;           // One of kEmitterType* constants above
    uint32_t  flags;          // Bitfield - see EmitterFlags
    Color3f   radiance;
    Point3f   position;       // Point emitters
    Vector3f  direction;      // Directional emitters
    int       attachedMeshId; // -1 unless area emitter; CPU also tracks this
};

enum EmitterFlags : uint32_t {
    EMITTER_FLAG_TWO_SIDED = 1u << 0,
};

// Minimal direction-sample record for NEE APIs.
struct EmitterDirectionSample {
    Point3f  p;          // Sampled position on / representing the emitter
    Vector3f d;          // Normalised direction from shading point to emitter
    float    dist;       // Distance to the sampled point (large value for directional)
    float    pdf;        // Directional PDF at the shading point
    bool     delta;      // True for delta emitters (point, directional)
    int      emitterId;  // Index of the sampled emitter

    HD EmitterDirectionSample()
        : p(0.f), d(0.f), dist(0.f), pdf(0.f), delta(false), emitterId(-1) {}
};

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------
struct Scene {
    Triangle*       triangles    = nullptr;
    uint32_t        triangleCount = 0;

    Material*       materials    = nullptr;
    uint32_t        materialCount = 0;

    MeshInstanceGPU* meshes      = nullptr;
    uint32_t        meshCount    = 0;

    EmitterGPU*     emitters     = nullptr;
    uint32_t        emitterCount = 0;

    BVH             bvh;
    bool            use_vertex_normals = false;

    // -----------------------------------------------------------------------
    // Intersection
    // -----------------------------------------------------------------------
    HD bool intersect(const Ray& ray, float t_min, float t_max,
                      SurfaceIntersection& rec) const
    {
        bool hit = false;

        if (bvh.nodeCount > 0) {
            hit = bvh.intersect(ray, t_min, t_max, triangles, rec, use_vertex_normals);
        } else {
            float closest = t_max;
            for (uint32_t i = 0; i < triangleCount; ++i) {
                SurfaceIntersection tmp;
                if (triangles[i].intersect(ray, t_min, closest, tmp, use_vertex_normals)) {
                    hit     = true;
                    closest = tmp.t;
                    rec     = tmp;
                }
            }
        }

        if (hit && rec.material_id >= 0 && rec.material_id < (int)materialCount) {
            const Material& mat = materials[rec.material_id];
            rec.albedo   = mat.albedo;
            rec.emission = mat.emission;
            rec.ior      = mat.intIOR;
            rec.mat_type = mat.type;
        }

        return hit;
    }

    HD int intersectAABBCount(const Ray& ray, float t_min, float t_max) const {
        if (bvh.nodeCount > 0)
            return bvh.intersectAABBCount(ray, t_min, t_max);
        return 0; // Un-accelerated scene has no AABB hierarchy.
    }

    // -----------------------------------------------------------------------
    // Emitter evaluation (device side)
    // -----------------------------------------------------------------------

    // Evaluate radiance of a specific emitter at a surface intersection.
    HD Color3f emitter_eval(int emitterId, const SurfaceIntersection& si) const {
        if (emitters == nullptr) return Color3f(0.f);
        if (emitterId < 0 || (uint32_t)emitterId >= emitterCount) return Color3f(0.f);

        const EmitterGPU& e = emitters[emitterId];

        if (e.type == kEmitterTypeArea) {
            const bool twoSided = (e.flags & EMITTER_FLAG_TWO_SIDED) != 0u;
            if (!twoSided && !si.front_face)
                return Color3f(0.f);
            return e.radiance;
        }

        // Point / directional / env - return stored radiance as a placeholder
        // until those emitter types are fully implemented.
        (void)si;
        return e.radiance;
    }

    // Resolve emitted radiance at an intersection by consulting the mesh's
    // attached emitter. Falls back to the emission baked into the material
    // if no emitter record is available.
    HD Color3f eval_surface_emission(const SurfaceIntersection& si) const {
        if (meshes == nullptr ||
            si.shape_id < 0 ||
            (uint32_t)si.shape_id >= meshCount)
        {
            return si.emission; // fallback: material-baked emission
        }

        const int meshEmitterId = meshes[si.shape_id].emitterId;
        if (meshEmitterId < 0 || (uint32_t)meshEmitterId >= emitterCount)
            return si.emission;

        return emitter_eval(meshEmitterId, si);
    }

    // -----------------------------------------------------------------------
    // NEE / direct-lighting API stubs (to be implemented).
    // -----------------------------------------------------------------------
    HD bool sample_emitter_direction(const SurfaceIntersection& /*si*/,
                                     const Point2f&              /*sample*/,
                                     EmitterDirectionSample&     ds,
                                     Color3f&                    weight) const
    {
        ds     = EmitterDirectionSample();
        weight = Color3f(0.f);
        return false;
    }

    HD float pdf_emitter_direction(const SurfaceIntersection&    /*si*/,
                                   const EmitterDirectionSample& /*ds*/) const
    {
        return 0.f;
    }

    HD Color3f eval_emitter_direction(const SurfaceIntersection&    /*si*/,
                                      const EmitterDirectionSample& /*ds*/) const
    {
        return Color3f(0.f);
    }

    // -----------------------------------------------------------------------
    // Host-side resource management
    // -----------------------------------------------------------------------

    void setTriangles(const Triangle* hostTriangles, uint32_t count) {
        if (triangles != nullptr) {
            CUDA_CHECK(cudaFree(triangles));
            triangles = nullptr;
        }
        triangleCount = count;
        if (triangleCount == 0) return;

        CUDA_CHECK(cudaMalloc(&triangles, triangleCount * sizeof(Triangle)));
        CUDA_CHECK(cudaMemcpy(triangles, hostTriangles,
                              triangleCount * sizeof(Triangle),
                              cudaMemcpyHostToDevice));
        bvh.build(hostTriangles, triangleCount);
    }

    void setMaterials(const Material* hostMaterials, uint32_t count) {
        if (materials != nullptr) {
            CUDA_CHECK(cudaFree(materials));
            materials = nullptr;
        }
        materialCount = count;
        if (materialCount == 0) return;

        CUDA_CHECK(cudaMalloc(&materials, materialCount * sizeof(Material)));
        CUDA_CHECK(cudaMemcpy(materials, hostMaterials,
                              materialCount * sizeof(Material),
                              cudaMemcpyHostToDevice));
    }

    void setMeshes(const MeshInstanceGPU* hostMeshes, uint32_t count) {
        if (meshes != nullptr) {
            CUDA_CHECK(cudaFree(meshes));
            meshes = nullptr;
        }
        meshCount = count;
        if (meshCount == 0) return;

        CUDA_CHECK(cudaMalloc(&meshes, meshCount * sizeof(MeshInstanceGPU)));
        CUDA_CHECK(cudaMemcpy(meshes, hostMeshes,
                              meshCount * sizeof(MeshInstanceGPU),
                              cudaMemcpyHostToDevice));
    }

    void setEmitters(const EmitterGPU* hostEmitters, uint32_t count) {
        if (emitters != nullptr) {
            CUDA_CHECK(cudaFree(emitters));
            emitters = nullptr;
        }
        emitterCount = count;
        if (emitterCount == 0) return;

        CUDA_CHECK(cudaMalloc(&emitters, emitterCount * sizeof(EmitterGPU)));
        CUDA_CHECK(cudaMemcpy(emitters, hostEmitters,
                              emitterCount * sizeof(EmitterGPU),
                              cudaMemcpyHostToDevice));
    }

    void clear() {
        if (triangles != nullptr) { CUDA_CHECK(cudaFree(triangles)); triangles = nullptr; }
        triangleCount = 0;
        bvh.clear();

        if (materials != nullptr) { CUDA_CHECK(cudaFree(materials)); materials = nullptr; }
        materialCount = 0;

        if (meshes != nullptr) { CUDA_CHECK(cudaFree(meshes)); meshes = nullptr; }
        meshCount = 0;

        if (emitters != nullptr) { CUDA_CHECK(cudaFree(emitters)); emitters = nullptr; }
        emitterCount = 0;
    }
};

} // namespace futaba
