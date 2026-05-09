#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <numeric>

#include "types.cuh"
#include "ray.cuh"
#include "triangle.cuh"
#include "surface_interaction.cuh"

#include <optix.h>

// Switch to toggle between OptiX hardware BVH and custom software BVH.
// Set to 0 to force the entire renderer to use the software BVH, ensuring 
// the path tracer and heatmap are evaluating the exact same AABBs.
#define FUTABA_USE_OPTIX 1

namespace futaba {

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

struct AABB {
    Point3f minP;
    Point3f maxP;

    HD AABB()
        : minP(1e30f),
          maxP(-1e30f) {}

    HD void expand(const Point3f& p) {
        minP.x = minP.x < p.x ? minP.x : p.x;
        minP.y = minP.y < p.y ? minP.y : p.y;
        minP.z = minP.z < p.z ? minP.z : p.z;
        maxP.x = maxP.x > p.x ? maxP.x : p.x;
        maxP.y = maxP.y > p.y ? maxP.y : p.y;
        maxP.z = maxP.z > p.z ? maxP.z : p.z;
    }

    HD void expand(const AABB& b) {
        expand(b.minP);
        expand(b.maxP);
    }

    HD Vector3f extent() const {
        return maxP - minP;
    }

    HD Point3f centroid() const {
        return Point3f(0.5f * (minP.x + maxP.x),
                       0.5f * (minP.y + maxP.y),
                       0.5f * (minP.z + maxP.z));
    }

    // Your exact, safe intersection logic. Safely swallows NaNs.
    HD bool intersect(const Ray& ray, float tMin, float tMax) const {
        float tx1 = (minP.x - ray.o.x) * ray.dRcp.x;
        float tx2 = (maxP.x - ray.o.x) * ray.dRcp.x;
        float tNear = tx1 < tx2 ? tx1 : tx2;
        float tFar = tx1 > tx2 ? tx1 : tx2;
        tMin = tNear > tMin ? tNear : tMin;
        tMax = tFar < tMax ? tFar : tMax;

        float ty1 = (minP.y - ray.o.y) * ray.dRcp.y;
        float ty2 = (maxP.y - ray.o.y) * ray.dRcp.y;
        tNear = ty1 < ty2 ? ty1 : ty2;
        tFar = ty1 > ty2 ? ty1 : ty2;
        tMin = tNear > tMin ? tNear : tMin;
        tMax = tFar < tMax ? tFar : tMax;

        float tz1 = (minP.z - ray.o.z) * ray.dRcp.z;
        float tz2 = (maxP.z - ray.o.z) * ray.dRcp.z;
        tNear = tz1 < tz2 ? tz1 : tz2;
        tFar = tz1 > tz2 ? tz1 : tz2;
        tMin = tNear > tMin ? tNear : tMin;
        tMax = tFar < tMax ? tFar : tMax;

        return tMax >= tMin;
    }

    HD bool intersectDist(const Ray& ray, float tMin, float tMax, float& dist) const {
        float tx1 = (minP.x - ray.o.x) * ray.dRcp.x;
        float tx2 = (maxP.x - ray.o.x) * ray.dRcp.x;
        float tNear = tx1 < tx2 ? tx1 : tx2;
        float tFar = tx1 > tx2 ? tx1 : tx2;
        tMin = tNear > tMin ? tNear : tMin;
        tMax = tFar < tMax ? tFar : tMax;

        float ty1 = (minP.y - ray.o.y) * ray.dRcp.y;
        float ty2 = (maxP.y - ray.o.y) * ray.dRcp.y;
        tNear = ty1 < ty2 ? ty1 : ty2;
        tFar = ty1 > ty2 ? ty1 : ty2;
        tMin = tNear > tMin ? tNear : tMin;
        tMax = tFar < tMax ? tFar : tMax;

        float tz1 = (minP.z - ray.o.z) * ray.dRcp.z;
        float tz2 = (maxP.z - ray.o.z) * ray.dRcp.z;
        tNear = tz1 < tz2 ? tz1 : tz2;
        tFar = tz1 > tz2 ? tz1 : tz2;
        tMin = tNear > tMin ? tNear : tMin;
        tMax = tFar < tMax ? tFar : tMax;

        if (tMax >= tMin) {
            dist = tMin;
            return true;
        }
        return false;
    }
};

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

    static AABB triangleBounds(const Triangle& t) {
        AABB b;
        b.expand(t.p0);
        b.expand(t.p1);
        b.expand(t.p2);
        return b;
    }

    static Point3f triangleCentroid(const Triangle& t) {
        return Point3f((t.p0.x + t.p1.x + t.p2.x) / 3.0f,
                       (t.p0.y + t.p1.y + t.p2.y) / 3.0f,
                       (t.p0.z + t.p1.z + t.p2.z) / 3.0f);
    }

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
                bounds.expand(triangleBounds(tri));
                centroidBounds.expand(triangleCentroid(tri));
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

            float splitPos = axis == 0 ? centroidBounds.centroid().x
                           : axis == 1 ? centroidBounds.centroid().y
                                       : centroidBounds.centroid().z;

            auto midIt = std::partition(hostIndices.begin() + start,
                                        hostIndices.begin() + start + count,
                                        [&](int triIdx) {
                                            Point3f c = triangleCentroid(hostTriangles[triIdx]);
                                            float value = axis == 0 ? c.x : (axis == 1 ? c.y : c.z);
                                            return value < splitPos;
                                        });

            int leftCount = (int)(midIt - (hostIndices.begin() + start));
            
            if (leftCount <= 0 || leftCount >= count) {
                leftCount = count / 2;
                std::nth_element(hostIndices.begin() + start,
                                 hostIndices.begin() + start + leftCount,
                                 hostIndices.begin() + start + count,[&](int a, int b) {
                                     Point3f ca = triangleCentroid(hostTriangles[a]);
                                     Point3f cb = triangleCentroid(hostTriangles[b]);
                                     float va = axis == 0 ? ca.x : (axis == 1 ? ca.y : ca.z);
                                     float vb = axis == 0 ? cb.x : (axis == 1 ? cb.y : cb.z);
                                     return va < vb;
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
                   1,
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

        int stack[64];
        int stackSize = 0;

        float dummyDist;
        if (!bvhNodes[0].bounds.intersectDist(ray, tMin, closest, dummyDist)) {
            return false;
        }

        stack[stackSize++] = 0;

        while (stackSize > 0) {
            int nodeIdx = stack[--stackSize];
            const BVHNode& node = bvhNodes[nodeIdx];

            if (node.isLeaf()) {
                for (int i = 0; i < node.triCount; ++i) {
                    int triIdx = bvhTriIndices[node.leftFirst + i];
                    SurfaceIntersection tmp;
                    if (triangles[triIdx].intersect(ray, tMin, closest, tmp, use_vertex_normals)) {
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
                    if (stackSize + 2 <= 64) {
                        if (distLeft < distRight) {
                            stack[stackSize++] = rightIdx;
                            stack[stackSize++] = leftIdx;
                        } else {
                            stack[stackSize++] = leftIdx;
                            stack[stackSize++] = rightIdx;
                        }
                    }
                } else if (hitLeft) {
                    if (stackSize + 1 <= 64) stack[stackSize++] = leftIdx;
                } else if (hitRight) {
                    if (stackSize + 1 <= 64) stack[stackSize++] = rightIdx;
                }
            }
        }

        return hit;
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

        int stack[64];
        int stackSize = 0;

        float dummyDist;
        aabb_tests++;
        if (!bvhNodes[0].bounds.intersectDist(ray, tMin, tMax, dummyDist)) {
            return aabb_tests;
        }

        stack[stackSize++] = 0;

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
                    if (stackSize + 2 <= 64) {
                        if (distLeft < distRight) {
                            stack[stackSize++] = rightIdx;
                            stack[stackSize++] = leftIdx;
                        } else {
                            stack[stackSize++] = leftIdx;
                            stack[stackSize++] = rightIdx;
                        }
                    }
                } else if (hitLeft) {
                    if (stackSize + 1 <= 64) stack[stackSize++] = leftIdx;
                } else if (hitRight) {
                    if (stackSize + 1 <= 64) stack[stackSize++] = rightIdx;
                }
            }
        }

        return aabb_tests;
    }
};

} // namespace futaba