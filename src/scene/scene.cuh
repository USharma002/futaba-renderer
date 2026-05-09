#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include "../shapes/triangle.cuh"
#include "../materials/material.cuh"
#include "../accel/bvh.cuh"

namespace futaba {

struct Scene {
    Triangle* triangles = nullptr;
    uint32_t triangleCount = 0;
    
    Material* materials = nullptr;
    uint32_t materialCount = 0;

    BVH bvh;
    bool use_vertex_normals = false;

    HD bool intersect(const Ray& ray, float t_min, float t_max, SurfaceIntersection& rec) const {
        bool hit = false;

        if (bvh.nodeCount > 0) {
            hit = bvh.intersect(ray, t_min, t_max, triangles, rec, use_vertex_normals);
        } else {
            float closest = t_max;
            for (uint32_t i = 0; i < triangleCount; ++i) {
                SurfaceIntersection temp;
                if (triangles[i].intersect(ray, t_min, closest, temp, use_vertex_normals)) {
                    hit = true;
                    closest = temp.t;
                    rec = temp;
                }
            }
        }

        if (hit && rec.material_id >= 0 && rec.material_id < (int)materialCount) {
            const Material& mat = materials[rec.material_id];
            rec.albedo = mat.albedo;
            rec.emission = mat.emission;
            rec.ior = mat.intIOR;
            // Store material type directly - will dispatch in SurfaceIntersection::sample_bsdf()
            rec.mat_type = mat.type;
        }

        return hit;
    }

    HD int intersectAABBCount(const Ray& ray, float t_min, float t_max) const {
        if (bvh.nodeCount > 0) {
            return bvh.intersectAABBCount(ray, t_min, t_max);
        } else {
            // Un-accelerated scene has 0 AABBs.
            return 0; 
        }
    }

    void setTriangles(const Triangle* hostTriangles, uint32_t count) {
        if (triangles != nullptr) {
            cudaFree(triangles);
        }
        triangleCount = count;
        if (triangleCount == 0) {
            triangles = nullptr;
            return;
        }

        cudaMalloc(&triangles, triangleCount * sizeof(Triangle));
        cudaMemcpy(triangles, hostTriangles, triangleCount * sizeof(Triangle), cudaMemcpyHostToDevice);
        bvh.build(hostTriangles, triangleCount);
    }
    
    void setMaterials(const Material* hostMaterials, uint32_t count) {
        if (materials != nullptr) {
            cudaFree(materials);
        }
        materialCount = count;
        if (materialCount == 0) {
            materials = nullptr;
            return;
        }

        cudaMalloc(&materials, materialCount * sizeof(Material));
        cudaMemcpy(materials, hostMaterials, materialCount * sizeof(Material), cudaMemcpyHostToDevice);
    }

    void clear() {
        if (triangles != nullptr) {
            cudaFree(triangles);
            triangles = nullptr;
        }
        triangleCount = 0;

        bvh.clear();

        if (materials != nullptr) {
            cudaFree(materials);
            materials = nullptr;
        }
        materialCount = 0;
    }
};

} // namespace futaba
