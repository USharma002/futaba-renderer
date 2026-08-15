#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include "common.cuh"
#include "triangle.cuh"
#include "material.cuh"
#include "bvh.cuh"
#include "envmap.cuh"
#include "distribution.cuh"
#include "emitter.cuh"
#include "mesh.cuh"
#include "scene_loader.h"

FUTABA_NAMESPACE_BEGIN

inline bool supportsPrefetch() {
    static int supported = -1;
    if (supported == -1) {
        int deviceId = 0;
        if (cudaGetDevice(&deviceId) == cudaSuccess) {
            int val = 0;
            if (cudaDeviceGetAttribute(&val, cudaDevAttrConcurrentManagedAccess, deviceId) == cudaSuccess) {
                supported = val;
            } else {
                supported = 0;
            }
        } else {
            supported = 0;
        }
    }
    return supported == 1;
}

inline void safeMemPrefetchAsync(const void* devPtr, size_t count, int deviceId) {
    if (supportsPrefetch()) {
        CUDA_CHECK(cudaMemPrefetchAsync(devPtr, count, deviceId));
    }
}

// Frees a CUDA-managed allocation if one is set, and nulls the pointer.
// Every Scene::set*() / clear() below uses this
template<typename T>
inline void freeManaged(T*& devPtr) {
    if (devPtr != nullptr) {
        CUDA_CHECK(cudaFree(devPtr));
        devPtr = nullptr;
    }
}

// Replaces a CUDA-managed device array with a fresh copy of hostPtr[0..count).
// Frees any existing allocation first, then allocates + copies + prefetches
// the new one. Every Scene::set*() below is a one-line call to this.
template<typename T>
inline void uploadManaged(T*& devPtr, const T* hostPtr, size_t count) {
    freeManaged(devPtr);
    if (count == 0 || hostPtr == nullptr)
        return;

    int deviceId = 0;
    cudaGetDevice(&deviceId);

    CUDA_CHECK(cudaMallocManaged(&devPtr, count * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(devPtr, hostPtr, count * sizeof(T), cudaMemcpyHostToDevice));
    safeMemPrefetchAsync(devPtr, count * sizeof(T), deviceId);
}


// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------
struct Scene {
    Triangle*       triangles    = nullptr;
    Material*       materials    = nullptr;
    MeshInstance*   meshes       = nullptr;
    EmitterInstance* emitters     = nullptr;
    Medium*         media        = nullptr;

    float*          emitterTriangleCdf = nullptr;
    int*            emissiveTriangleIndices = nullptr; // maps cdf index -> global triangle index
    int*            emissiveGlobalToIndex = nullptr;   // maps global triangle idx -> emissive array idx or -1

    EnvironmentMapEmitter envMap;

    BVH             bvh;

    uint32_t        triangleCount = 0;
    uint32_t        materialCount = 0;
    uint32_t        meshCount    = 0;
    uint32_t        emitterCount = 0;
    uint32_t        mediumCount  = 0;
    int             emissiveTriCount = 0;
    float           emitterTriangleFuncSum = 0.f; // sum of weights (area * intensity)

    Point3f         boundsMin = Point3f(1e30f, 1e30f, 1e30f);
    Point3f         boundsMax = Point3f(-1e30f, -1e30f, -1e30f);

    bool            use_vertex_normals = false;
    bool            use_nee = true;

    // -----------------------------------------------------------------------
    // Intersection
    // -----------------------------------------------------------------------
    HD bool intersect(const Ray& ray, float t_min, float t_max,
                      SurfaceInteraction& rec) const
    {
        bool hit = false;

        if (bvh.nodeCount > 0) {
            hit = bvh.intersect(ray, t_min, t_max, triangles, rec, use_vertex_normals);
        } else {
            float closest = t_max;
            for (uint32_t i = 0; i < triangleCount; ++i) {
                SurfaceInteraction tmp;
                if (triangles[i].intersect(ray, t_min, closest, tmp, use_vertex_normals, (int)i)) {
                    hit     = true;
                    closest = tmp.t;
                    rec     = tmp;
                }
            }
        }

        if (hit && materials != nullptr && rec.material_id >= 0 && (uint32_t)rec.material_id < materialCount) {
            const Material& mat = materials[rec.material_id];
            if (rec.primitive_id >= 0 && (uint32_t)rec.primitive_id < triangleCount) {
                triangles[rec.primitive_id].apply_normal_map(mat, rec);
            }
        }

        return hit;
    }

    HD bool occluded(const Ray& ray, float t_min, float t_max, int target_mesh_id = -1) const {
        if (bvh.nodeCount > 0)
            return bvh.occluded(ray, t_min, t_max, target_mesh_id, triangles, materials, materialCount);

        for (uint32_t i = 0; i < triangleCount; ++i) {
            const Triangle& tri = triangles[i];
            if (tri.mesh_id == target_mesh_id) continue;
            if (tri.material_id >= 0 && tri.material_id < (int)materialCount) {
                int mat_type = materials[tri.material_id].type;
                if (Material::isShadowTransparent((BSDFType)mat_type)) {
                    continue;
                }
            }
            SurfaceInteraction tmp;
            if (triangles[i].intersect(ray, t_min, t_max, tmp, false, (int)i))
                return true;
        }
        return false;
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
    HD Color3f emitter_eval(int emitterId, const SurfaceInteraction& si) const {
        if (emitters == nullptr) return Color3f(0.f);
        if (emitterId < 0 || (uint32_t)emitterId >= emitterCount) return Color3f(0.f);

        const EmitterInstance& e = emitters[emitterId];

        if (e.type == EmitterType::Area) {
            if (!e.twoSided && !si.front_face)
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
    HD Color3f eval_surface_emission(const SurfaceInteraction& si) const {
        if (materials != nullptr && si.material_id >= 0 && (uint32_t)si.material_id < materialCount) {
            if (meshes != nullptr && si.shape_id >= 0 && (uint32_t)si.shape_id < meshCount) {
                const int meshEmitterId = meshes[si.shape_id].emitterId;
                if (meshEmitterId >= 0 && (uint32_t)meshEmitterId < emitterCount) {
                    return emitter_eval(meshEmitterId, si);
                }
            }
            return materials[si.material_id].emission;
        }
        return Color3f(0.f);
    }

    HD Color3f eval_environment(const Vector3f& dirWorld) const {
        return envMap.eval(dirWorld);
    }

    void setEmitterTriangleDistribution(const float* hostCdf,
                                        int cdfCount,
                                        float funcSum,
                                        const int* hostEmissiveTriangleIndices,
                                        int emissiveCount,
                                        const int* hostEmissiveGlobalToIndex,
                                        int globalToIndexCount)
    {
        emissiveTriCount = emissiveCount;
        emitterTriangleFuncSum = funcSum;

        uploadManaged(emitterTriangleCdf, hostCdf, cdfCount > 0 ? (size_t)cdfCount : 0);
        uploadManaged(emissiveTriangleIndices, hostEmissiveTriangleIndices, emissiveCount > 0 ? (size_t)emissiveCount : 0);
        uploadManaged(emissiveGlobalToIndex, hostEmissiveGlobalToIndex, globalToIndexCount > 0 ? (size_t)globalToIndexCount : 0);
    }

    // -----------------------------------------------------------------------
    // Host-side resource management
    // -----------------------------------------------------------------------

    void setTriangles(const Triangle* hostTriangles, uint32_t count) {
        triangleCount = count;
        if (triangleCount == 0) {
            freeManaged(triangles);
            return;
        }

        // Compute host-side scene bounds for normalization heuristics.
        AABB bounds;
        for (uint32_t i = 0; i < triangleCount; ++i) {
            bounds.expand(hostTriangles[i].bounds());
        }
        boundsMin = bounds.min;
        boundsMax = bounds.max;

        uploadManaged(triangles, hostTriangles, (size_t)triangleCount);
        bvh.build(triangles, triangleCount);
    }

    void setMaterials(const Material* hostMaterials, uint32_t count) {
        materialCount = count;
        uploadManaged(materials, hostMaterials, (size_t)count);
    }

    void setMeshes(const MeshInstance* hostMeshes, uint32_t count) {
        meshCount = count;
        uploadManaged(meshes, hostMeshes, (size_t)count);
    }

    void setEmitters(const EmitterInstance* hostEmitters, uint32_t count) {
        emitterCount = count;
        uploadManaged(emitters, hostEmitters, (size_t)count);
    }

    void setMedia(const Medium* hostMedia, uint32_t count) {
        mediumCount = count;
        uploadManaged(media, hostMedia, (size_t)count);
    }

    void setEnvironmentMap(const Color3f* hostPixels, uint32_t width, uint32_t height,
                           const Matrix4f& envToWorld) {
        envMap.setMap(hostPixels, width, height, envToWorld);
    }

    void setConstantEnvironment(const Color3f& radiance) {
        envMap.setConstant(radiance);
    }

    void load(const CPUScene& cpuScene) {
        clear();

        // Triangles & BVH
        setTriangles(cpuScene.triangles.data(), (uint32_t)cpuScene.triangles.size());

        // Materials
        setMaterials(cpuScene.materials.data(), (uint32_t)cpuScene.materials.size());

        // Meshes
        setMeshes(cpuScene.meshes.data(), (uint32_t)cpuScene.meshes.size());

        // Emitters
        setEmitters(cpuScene.emitters.data(), (uint32_t)cpuScene.emitters.size());

        // Participating Media
        setMedia(cpuScene.media.data(), (uint32_t)cpuScene.media.size());

        // Environment Map
        if (cpuScene.envMap.hasEnvMap) {
            setEnvironmentMap(cpuScene.envMap.pixels.data(), 
                              cpuScene.envMap.width, 
                              cpuScene.envMap.height, 
                              cpuScene.envMap.toWorld);
        } else if (cpuScene.envMap.hasConstant) {
            setConstantEnvironment(cpuScene.envMap.constantRadiance);
        }

        // Emissive triangle distribution
        std::vector<int> emissiveTriIndices;
        std::vector<float> emissiveWeights;
        emissiveTriIndices.reserve(cpuScene.triangles.size());
        emissiveWeights.reserve(cpuScene.triangles.size());

        auto triangleEmission = [&](const Triangle &t) -> Color3f {
            if (t.mesh_id >= 0 && (uint32_t)t.mesh_id < cpuScene.meshes.size()) {
                const int emitterId = cpuScene.meshes[t.mesh_id].emitterId;
                if (emitterId >= 0 && (uint32_t)emitterId < cpuScene.emitters.size()) {
                    return cpuScene.emitters[emitterId].radiance;
                }
            }
            if (t.material_id >= 0 && (uint32_t)t.material_id < cpuScene.materials.size()) {
                return cpuScene.materials[t.material_id].emission;
            }
            return Color3f(0.f);
        };

        for (size_t i = 0; i < cpuScene.triangles.size(); ++i) {
            const Triangle &t = cpuScene.triangles[i];
            float area = 0.5f * length(cross(t.p1 - t.p0, t.p2 - t.p0));
            Color3f emission = triangleEmission(t);
            float w = area * getLuminance(emission);
            if (w > 0.f) {
                emissiveTriIndices.push_back((int)i);
                emissiveWeights.push_back(w);
            }
        }

        Distribution1D lightDist;
        std::vector<int> globalToIndex(cpuScene.triangles.size(), -1);
        if (!emissiveWeights.empty()) {
            lightDist.build(emissiveWeights);
            for (size_t i = 0; i < emissiveTriIndices.size(); ++i) {
                globalToIndex[emissiveTriIndices[i]] = (int)i;
            }
        }

        setEmitterTriangleDistribution(lightDist.cdf.data(), (int)lightDist.cdf.size(), lightDist.funcSum,
                                       emissiveTriIndices.data(), (int)emissiveTriIndices.size(),
                                       globalToIndex.data(), (int)globalToIndex.size());

    }

    void clear() {
        freeManaged(triangles);
        triangleCount = 0;
        bvh.clear();

        freeManaged(materials);
        materialCount = 0;

        freeManaged(meshes);
        meshCount = 0;

        freeManaged(emitters);
        emitterCount = 0;

        freeManaged(media);
        mediumCount = 0;

        freeManaged(emitterTriangleCdf);
        freeManaged(emissiveTriangleIndices);
        freeManaged(emissiveGlobalToIndex);

        emissiveTriCount = 0;
        emitterTriangleFuncSum = 0.f;

        envMap.clear();
    }
};

FUTABA_NAMESPACE_END