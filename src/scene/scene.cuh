#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include "triangle.cuh"
#include "material.cuh"
#include "bvh.cuh"
#include "envmap.cuh"
#include "distribution.cuh"
#include "emitter.cuh"
#include "mesh.cuh"
#include "medium.cuh"

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


// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------
struct Scene {
    Triangle*       triangles    = nullptr;
    Material*       materials    = nullptr;
    MeshInstanceGPU* meshes      = nullptr;
    EmitterGPU*     emitters     = nullptr;

    float*          emitterTriangleCdf = nullptr;
    int*            emissiveTriangleIndices = nullptr; // maps cdf index -> global triangle index
    int*            emissiveGlobalToIndex = nullptr;   // maps global triangle idx -> emissive array idx or -1
    int*            nonAreaEmitterIndices = nullptr;   // device array of indices into emitters[]

    EnvironmentMapEmitter envMap;

    BVH             bvh;

    uint32_t        triangleCount = 0;
    uint32_t        materialCount = 0;
    uint32_t        meshCount    = 0;
    uint32_t        emitterCount = 0;
    int             emissiveTriCount = 0;
    int             nonAreaEmitterCount = 0;
    float           emitterTriangleFuncSum = 0.f; // sum of weights (area * intensity)

    Point3f         boundsMin = Point3f(1e30f, 1e30f, 1e30f);
    Point3f         boundsMax = Point3f(-1e30f, -1e30f, -1e30f);

    bool            use_vertex_normals = false;
    bool            use_nee = true;

    bool            hasMedium = false;
    int             mediumMeshId = -1;
    Medium          medium;

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
                if (triangles[i].intersect(ray, t_min, closest, tmp, use_vertex_normals, (int)i)) {
                    hit     = true;
                    closest = tmp.t;
                    rec     = tmp;
                }
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
                if (mat_type == BSDF_ID_NULL || mat_type == BSDF_ID_THINDIELECTRIC) {
                    continue;
                }
            }
            SurfaceIntersection tmp;
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
        if (emitterTriangleCdf != nullptr) {
            CUDA_CHECK(cudaFree(emitterTriangleCdf));
            emitterTriangleCdf = nullptr;
        }
        if (emissiveTriangleIndices != nullptr) {
            CUDA_CHECK(cudaFree(emissiveTriangleIndices));
            emissiveTriangleIndices = nullptr;
        }
        if (emissiveGlobalToIndex != nullptr) {
            CUDA_CHECK(cudaFree(emissiveGlobalToIndex));
            emissiveGlobalToIndex = nullptr;
        }

        emissiveTriCount = emissiveCount;
        emitterTriangleFuncSum = funcSum;

        int deviceId = 0;
        cudaGetDevice(&deviceId);

        if (cdfCount > 0 && hostCdf != nullptr) {
            CUDA_CHECK(cudaMallocManaged(&emitterTriangleCdf, (size_t)cdfCount * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(emitterTriangleCdf, hostCdf, (size_t)cdfCount * sizeof(float), cudaMemcpyHostToDevice));
            safeMemPrefetchAsync(emitterTriangleCdf, (size_t)cdfCount * sizeof(float), deviceId);
        }

        if (emissiveCount > 0 && hostEmissiveTriangleIndices != nullptr) {
            CUDA_CHECK(cudaMallocManaged(&emissiveTriangleIndices, (size_t)emissiveCount * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(emissiveTriangleIndices, hostEmissiveTriangleIndices, (size_t)emissiveCount * sizeof(int), cudaMemcpyHostToDevice));
            safeMemPrefetchAsync(emissiveTriangleIndices, (size_t)emissiveCount * sizeof(int), deviceId);
        }

        if (globalToIndexCount > 0 && hostEmissiveGlobalToIndex != nullptr) {
            CUDA_CHECK(cudaMallocManaged(&emissiveGlobalToIndex, (size_t)globalToIndexCount * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(emissiveGlobalToIndex, hostEmissiveGlobalToIndex, (size_t)globalToIndexCount * sizeof(int), cudaMemcpyHostToDevice));
            safeMemPrefetchAsync(emissiveGlobalToIndex, (size_t)globalToIndexCount * sizeof(int), deviceId);
        }
    }

    void setNonAreaEmitters(const int* hostEmitterIndices, int count) {
        if (nonAreaEmitterIndices != nullptr) {
            CUDA_CHECK(cudaFree(nonAreaEmitterIndices));
            nonAreaEmitterIndices = nullptr;
        }

        nonAreaEmitterCount = count;
        if (count <= 0 || hostEmitterIndices == nullptr)
            return;

        int deviceId = 0;
        cudaGetDevice(&deviceId);

        CUDA_CHECK(cudaMallocManaged(&nonAreaEmitterIndices, (size_t)count * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(nonAreaEmitterIndices, hostEmitterIndices, (size_t)count * sizeof(int), cudaMemcpyHostToDevice));
        safeMemPrefetchAsync(nonAreaEmitterIndices, (size_t)count * sizeof(int), deviceId);
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

        // Compute host-side scene bounds for normalization heuristics.
        Point3f minP(1e30f, 1e30f, 1e30f);
        Point3f maxP(-1e30f, -1e30f, -1e30f);
        for (uint32_t i = 0; i < triangleCount; ++i) {
            const Triangle& t = hostTriangles[i];
            minP.x = fminf(minP.x, fminf(t.p0.x, fminf(t.p1.x, t.p2.x)));
            minP.y = fminf(minP.y, fminf(t.p0.y, fminf(t.p1.y, t.p2.y)));
            minP.z = fminf(minP.z, fminf(t.p0.z, fminf(t.p1.z, t.p2.z)));
            maxP.x = fmaxf(maxP.x, fmaxf(t.p0.x, fmaxf(t.p1.x, t.p2.x)));
            maxP.y = fmaxf(maxP.y, fmaxf(t.p0.y, fmaxf(t.p1.y, t.p2.y)));
            maxP.z = fmaxf(maxP.z, fmaxf(t.p0.z, fmaxf(t.p1.z, t.p2.z)));
        }
        boundsMin = minP;
        boundsMax = maxP;

        int deviceId = 0;
        cudaGetDevice(&deviceId);

        CUDA_CHECK(cudaMallocManaged(&triangles, triangleCount * sizeof(Triangle)));
        CUDA_CHECK(cudaMemcpy(triangles, hostTriangles, triangleCount * sizeof(Triangle), cudaMemcpyHostToDevice));
        safeMemPrefetchAsync(triangles, triangleCount * sizeof(Triangle), deviceId);
        
        bvh.build(triangles, triangleCount);
    }

    void setMaterials(const Material* hostMaterials, uint32_t count) {
        if (materials != nullptr) {
            CUDA_CHECK(cudaFree(materials));
            materials = nullptr;
        }
        materialCount = count;
        if (materialCount == 0) return;

        int deviceId = 0;
        cudaGetDevice(&deviceId);

        CUDA_CHECK(cudaMallocManaged(&materials, materialCount * sizeof(Material)));
        CUDA_CHECK(cudaMemcpy(materials, hostMaterials, materialCount * sizeof(Material), cudaMemcpyHostToDevice));
        safeMemPrefetchAsync(materials, materialCount * sizeof(Material), deviceId);
    }

    void setMeshes(const MeshInstanceGPU* hostMeshes, uint32_t count) {
        if (meshes != nullptr) {
            CUDA_CHECK(cudaFree(meshes));
            meshes = nullptr;
        }
        meshCount = count;
        if (meshCount == 0) return;

        int deviceId = 0;
        cudaGetDevice(&deviceId);

        CUDA_CHECK(cudaMallocManaged(&meshes, meshCount * sizeof(MeshInstanceGPU)));
        CUDA_CHECK(cudaMemcpy(meshes, hostMeshes, meshCount * sizeof(MeshInstanceGPU), cudaMemcpyHostToDevice));
        safeMemPrefetchAsync(meshes, meshCount * sizeof(MeshInstanceGPU), deviceId);
    }

    void setEmitters(const EmitterGPU* hostEmitters, uint32_t count) {
        if (emitters != nullptr) {
            CUDA_CHECK(cudaFree(emitters));
            emitters = nullptr;
        }
        emitterCount = count;
        if (emitterCount == 0) return;

        int deviceId = 0;
        cudaGetDevice(&deviceId);

        CUDA_CHECK(cudaMallocManaged(&emitters, emitterCount * sizeof(EmitterGPU)));
        CUDA_CHECK(cudaMemcpy(emitters, hostEmitters, emitterCount * sizeof(EmitterGPU), cudaMemcpyHostToDevice));
        safeMemPrefetchAsync(emitters, emitterCount * sizeof(EmitterGPU), deviceId);
    }

    void setEnvironmentMap(const Color3f* hostPixels, uint32_t width, uint32_t height,
                           const Matrix4f& envToWorld) {
        envMap.setMap(hostPixels, width, height, envToWorld);
    }

    void setConstantEnvironment(const Color3f& radiance) {
        envMap.setConstant(radiance);
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

        if (emitterTriangleCdf != nullptr) { CUDA_CHECK(cudaFree(emitterTriangleCdf)); emitterTriangleCdf = nullptr; }
        if (emissiveTriangleIndices != nullptr) { CUDA_CHECK(cudaFree(emissiveTriangleIndices)); emissiveTriangleIndices = nullptr; }
        if (emissiveGlobalToIndex != nullptr) { CUDA_CHECK(cudaFree(emissiveGlobalToIndex)); emissiveGlobalToIndex = nullptr; }
        if (nonAreaEmitterIndices != nullptr) { CUDA_CHECK(cudaFree(nonAreaEmitterIndices)); nonAreaEmitterIndices = nullptr; }

        emissiveTriCount = 0;
        nonAreaEmitterCount = 0;
        emitterTriangleFuncSum = 0.f;

        envMap.clear();
    }
};

} // namespace futaba
