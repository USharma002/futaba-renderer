#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <numeric>

#include "types.cuh"
#include "bbox.cuh"
#include "ray.cuh"
#include "triangle.cuh"
#include "surface_interaction.cuh"
#include "material.cuh"

#include <optix.h>

// Switch to toggle between OptiX hardware BVH and custom software BVH.
// Set to 0 to force the entire renderer to use the software BVH, ensuring 
// the path tracer and heatmap are evaluating the exact same AABBs.
#define FUTABA_USE_OPTIX 1

FUTABA_NAMESPACE_BEGIN

constexpr int kMaxTraversalStackSize = 64;

struct BVH;
void buildOptixBVH(BVH& bvh, const Triangle* hostTriangles, uint32_t triangleCount);
void clearOptixBVH(BVH& bvh);

template <typename T>
HD void packPointer(T* ptr, unsigned int& i0, unsigned int& i1) {
    unsigned long long packed = reinterpret_cast<unsigned long long>(ptr);
    i0 = static_cast<unsigned int>(packed & 0x00000000ffffffffULL);
    i1 = static_cast<unsigned int>(packed >> 32);
}

template <typename T>
HD T* unpackPointer(unsigned int i0, unsigned int i1) {
    unsigned long long packed = (static_cast<unsigned long long>(i1) << 32) | static_cast<unsigned long long>(i0);
    return reinterpret_cast<T*>(packed);
}

struct alignas(32) BVHNode {
    AABB bounds;
    int leftFirst = -1;
    int triCount = 0;

    HD bool isLeaf() const { return triCount > 0; }
};

struct BVH {
    BVHNode* nodes = nullptr;
    uint32_t nodeCount = 0;

    int* triIndices = nullptr;
    uint32_t triIndexCount = 0;

    OptixTraversableHandle traversable = 0;
    CUdeviceptr accelBuffer = 0;



    void clear() {
        if (nodes != nullptr) {
            cudaFree(nodes);
            nodes = nullptr;
        }
        nodeCount = 0;

        if (triIndices != nullptr) {
            cudaFree(triIndices);
            triIndices = nullptr;
        }
        triIndexCount = 0;

        clearOptixBVH(*this);
    }

    void build(const Triangle* hostTriangles, uint32_t triangleCount) {
        clear();
        triIndexCount = triangleCount;
        if (triangleCount == 0) {
            return;
        }

        std::vector<int> hostIndices(triangleCount);
        std::iota(hostIndices.begin(), hostIndices.end(), 0);

        std::vector<BVHNode> hostNodes;
        // Pre-allocate to prevent vector reallocations from breaking references
        hostNodes.reserve(triangleCount * 2);
        
        // Push Root Node
        hostNodes.push_back(BVHNode());

        auto buildRecursive = [&](auto&& self, int nodeIndex, int start, int count) -> void {
            AABB bounds;
            AABB centroidBounds;
            for (int i = start; i < start + count; ++i) {
                const Triangle& tri = hostTriangles[hostIndices[i]];
                bounds.expand(tri.bounds());
                centroidBounds.expand(tri.centroid());
            }
            
            // Assigning by array index is perfectly memory safe
            hostNodes[nodeIndex].bounds = bounds;

            if (count <= 4) {
                hostNodes[nodeIndex].leftFirst = start;
                hostNodes[nodeIndex].triCount = count;
                return;
            }

            auto ext = centroidBounds.extent();
            int axis = 0;
            if (ext.y > ext.x && ext.y >= ext.z) {
                axis = 1;
            } else if (ext.z > ext.x && ext.z > ext.y) {
                axis = 2;
            }

            float splitPos = centroidBounds.centroid()[axis];

            auto midIt = std::partition(hostIndices.begin() + start,
                                        hostIndices.begin() + start + count,
                                        [&](int triIdx) {
                                            Point3f c = hostTriangles[triIdx].centroid();
                                            return c[axis] < splitPos;
                                        });

            int leftCount = (int)(midIt - (hostIndices.begin() + start));
            
            if (leftCount <= 0 || leftCount >= count) {
                leftCount = count / 2;
                std::nth_element(hostIndices.begin() + start,
                                 hostIndices.begin() + start + leftCount,
                                 hostIndices.begin() + start + count,[&](int a, int b) {
                                     Point3f ca = hostTriangles[a].centroid();
                                     Point3f cb = hostTriangles[b].centroid();
                                     return ca[axis] < cb[axis];
                                 });
            }

            int rightCount = count - leftCount;

            // GPU Optimization: Allocate children completely contiguously! 
            // Because of this, Right Child index is ALWAYS (Left Child index + 1).
            int leftChildIdx = (int)hostNodes.size();
            hostNodes.push_back(BVHNode()); // Left Child
            hostNodes.push_back(BVHNode()); // Right Child

            hostNodes[nodeIndex].leftFirst = leftChildIdx;
            hostNodes[nodeIndex].triCount = 0;

            self(self, leftChildIdx, start, leftCount);
            self(self, leftChildIdx + 1, start + leftCount, rightCount);
        };

        // Start recursive build from Root Node (index 0)
        buildRecursive(buildRecursive, 0, 0, (int)triangleCount);

        nodeCount = (uint32_t)hostNodes.size();

        cudaMalloc(&nodes, nodeCount * sizeof(BVHNode));
        cudaMemcpy(nodes, hostNodes.data(), nodeCount * sizeof(BVHNode), cudaMemcpyHostToDevice);

        cudaMalloc(&triIndices, triIndexCount * sizeof(int));
        cudaMemcpy(triIndices, hostIndices.data(), triIndexCount * sizeof(int), cudaMemcpyHostToDevice);

        buildOptixBVH(*this, hostTriangles, triangleCount);
    }

    HD bool intersect(const Ray& ray,
                      float tMin,
                      float tMax,
                      const Triangle* __restrict__ triangles, // __restrict__ uses fast L1 texture cache
                      SurfaceIntersection& rec,
                      bool use_vertex_normals) const {
#if defined(FUTABA_OPTIX_DEVICE_PROGRAMS) && FUTABA_USE_OPTIX
        if (traversable == 0) {
            return false;
        }

        unsigned int packed0 = 0;
        unsigned int packed1 = 0;
        packPointer(&rec, packed0, packed1);

        optixTrace(traversable,
                   make_float3(ray.o.x, ray.o.y, ray.o.z),
                   make_float3(ray.d.x, ray.d.y, ray.d.z),
                   tMin,
                   tMax,
                   0.0f,
                   OptixVisibilityMask(255),
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                   0,
                   2,
                   0,
                   packed0,
                   packed1);

        return rec.is_valid();
#else
        const BVHNode* __restrict__ bvhNodes = nodes;
        const int* __restrict__ bvhTriIndices = triIndices;

        if (bvhNodes == nullptr || bvhTriIndices == nullptr || nodeCount == 0) {
            return false;
        }

        bool hit = false;
        float closest = tMax;

        int stack[kMaxTraversalStackSize];
        int stackSize = 0;

        float dummyDist;
        if (!bvhNodes[0].bounds.intersectDist(ray, tMin, closest, dummyDist)) {
            return false;
        }

        if (stackSize < kMaxTraversalStackSize) {
            stack[stackSize++] = 0;
        }

        while (stackSize > 0) {
            int nodeIdx = stack[--stackSize];
            const BVHNode& node = bvhNodes[nodeIdx];

            if (node.isLeaf()) {
                for (int i = 0; i < node.triCount; ++i) {
                    int triIdx = bvhTriIndices[node.leftFirst + i];
                    SurfaceIntersection tmp;
                    if (triangles[triIdx].intersect(ray, tMin, closest, tmp, use_vertex_normals, triIdx)) {
                        hit = true;
                        closest = tmp.t;
                        rec = tmp;
                    }
                }
            } else {
                int leftIdx = node.leftFirst;
                int rightIdx = leftIdx + 1;

                float distLeft, distRight;
                bool hitLeft = bvhNodes[leftIdx].bounds.intersectDist(ray, tMin, closest, distLeft);
                bool hitRight = bvhNodes[rightIdx].bounds.intersectDist(ray, tMin, closest, distRight);

                if (hitLeft && hitRight) {
                    if (stackSize + 2 <= kMaxTraversalStackSize) {
                        if (distLeft < distRight) {
                            stack[stackSize++] = rightIdx;
                            stack[stackSize++] = leftIdx;
                        } else {
                            stack[stackSize++] = leftIdx;
                            stack[stackSize++] = rightIdx;
                        }
                    }
                } else if (hitLeft) {
                    if (stackSize + 1 <= kMaxTraversalStackSize) stack[stackSize++] = leftIdx;
                } else if (hitRight) {
                    if (stackSize + 1 <= kMaxTraversalStackSize) stack[stackSize++] = rightIdx;
                }
            }
        }

        return hit;
#endif
    }

    // Shadow ray: returns true if any geometry occludes [tMin, tMax].
    HD bool occluded(const Ray& ray, float tMin, float tMax,
                     int target_mesh_id = -1,
                     const Triangle* __restrict__ triangles = nullptr,
                     const Material* __restrict__ materials = nullptr,
                     int materialCount = 0) const {
#if defined(FUTABA_OPTIX_DEVICE_PROGRAMS) && FUTABA_USE_OPTIX
        if (traversable == 0)
            return false;

        unsigned int target_mesh_id_u = (unsigned int)target_mesh_id;
        unsigned int hit = 0;
        optixTrace(traversable,
                   make_float3(ray.o.x, ray.o.y, ray.o.z),
                   make_float3(ray.d.x, ray.d.y, ray.d.z),
                   tMin,
                   tMax,
                   0.0f,
                   OptixVisibilityMask(255),
                   OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                   1,  // SBT offset (shadow ray type)
                   2,  // SBT stride (2 ray types)
                   1,  // miss SBT index (shadow miss)
                   target_mesh_id_u,
                   hit);
        return hit != 0;
#else
        const BVHNode* __restrict__ bvhNodes = nodes;
        const int* __restrict__ bvhTriIndices = triIndices;

        if (bvhNodes == nullptr || bvhTriIndices == nullptr || nodeCount == 0 || triangles == nullptr) {
            return false;
        }

        int stack[kMaxTraversalStackSize];
        int stackSize = 0;

        float dummyDist;
        if (!bvhNodes[0].bounds.intersectDist(ray, tMin, tMax, dummyDist)) {
            return false;
        }

        if (stackSize < kMaxTraversalStackSize) {
            stack[stackSize++] = 0;
        }

        while (stackSize > 0) {
            int nodeIdx = stack[--stackSize];
            const BVHNode& node = bvhNodes[nodeIdx];

            if (node.isLeaf()) {
                for (int i = 0; i < node.triCount; ++i) {
                    int triIdx = bvhTriIndices[node.leftFirst + i];
                    const Triangle& tri = triangles[triIdx];
                    if (tri.mesh_id == target_mesh_id) {
                        continue;
                    }
                    if (materials && tri.material_id >= 0 && tri.material_id < materialCount) {
                        int mat_type = materials[tri.material_id].type;
                        if (Material::isShadowTransparent((BSDFType)mat_type)) {
                            continue;
                        }
                    }
                    SurfaceIntersection tmp;
                    if (tri.intersect(ray, tMin, tMax, tmp, false, triIdx)) {
                        return true;
                    }
                }
            } else {
                int leftIdx = node.leftFirst;
                int rightIdx = leftIdx + 1;

                float distLeft, distRight;
                bool hitLeft = bvhNodes[leftIdx].bounds.intersectDist(ray, tMin, tMax, distLeft);
                bool hitRight = bvhNodes[rightIdx].bounds.intersectDist(ray, tMin, tMax, distRight);

                if (hitLeft && hitRight) {
                    if (stackSize + 2 <= kMaxTraversalStackSize) {
                        if (distLeft < distRight) {
                            stack[stackSize++] = rightIdx;
                            stack[stackSize++] = leftIdx;
                        } else {
                            stack[stackSize++] = leftIdx;
                            stack[stackSize++] = rightIdx;
                        }
                    }
                } else if (hitLeft) {
                    if (stackSize + 1 <= kMaxTraversalStackSize) stack[stackSize++] = leftIdx;
                } else if (hitRight) {
                    if (stackSize + 1 <= kMaxTraversalStackSize) stack[stackSize++] = rightIdx;
                }
            }
        }
        return false;
#endif
    }

    HD int intersectAABBCount(const Ray& ray,
                              float tMin,
                              float tMax) const {
        const BVHNode* __restrict__ bvhNodes = nodes;

        if (bvhNodes == nullptr || nodeCount == 0) {
            return 0;
        }

        int aabb_tests = 0;

        int stack[kMaxTraversalStackSize];
        int stackSize = 0;

        float dummyDist;
        aabb_tests++;
        if (!bvhNodes[0].bounds.intersectDist(ray, tMin, tMax, dummyDist)) {
            return aabb_tests;
        }

        if (stackSize < kMaxTraversalStackSize) {
            stack[stackSize++] = 0;
        }

        while (stackSize > 0) {
            int nodeIdx = stack[--stackSize];
            const BVHNode& node = bvhNodes[nodeIdx];

            if (!node.isLeaf()) {
                int leftIdx = node.leftFirst;
                int rightIdx = leftIdx + 1;

                float distLeft, distRight;
                aabb_tests += 2;
                bool hitLeft = bvhNodes[leftIdx].bounds.intersectDist(ray, tMin, tMax, distLeft);
                bool hitRight = bvhNodes[rightIdx].bounds.intersectDist(ray, tMin, tMax, distRight);

                if (hitLeft && hitRight) {
                    if (stackSize + 2 <= kMaxTraversalStackSize) {
                        if (distLeft < distRight) {
                            stack[stackSize++] = rightIdx;
                            stack[stackSize++] = leftIdx;
                        } else {
                            stack[stackSize++] = leftIdx;
                            stack[stackSize++] = rightIdx;
                        }
                    }
                } else if (hitLeft) {
                    if (stackSize + 1 <= kMaxTraversalStackSize) stack[stackSize++] = leftIdx;
                } else if (hitRight) {
                    if (stackSize + 1 <= kMaxTraversalStackSize) stack[stackSize++] = rightIdx;
                }
            }
        }

        return aabb_tests;
    }

    
};

FUTABA_NAMESPACE_END