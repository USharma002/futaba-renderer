#pragma once

#include "types.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"
#include "triangle.cuh"
#include "distribution.cuh"
#include "emitter_sample.cuh"

FUTABA_NAMESPACE_BEGIN

struct AreaEmitter {
    // Samples a point on the area lights in the scene
    HD bool sample(const Scene&              scene,
                   const SurfaceInteraction& ref,
                   const Point3f&             u,
                   const CDFLightSamplerData& data,
                   EmitterSample&             es) const
    {
        int   local_idx  = -1;
        float du         = 0.f;
        float weight_val = 0.f;
        sample1D_device(data.emitterTriangleCdf, data.emissiveTriCount,
                        data.emitterTriangleFuncSum, u.x,
                        local_idx, du, weight_val);
        if (local_idx < 0 || local_idx >= data.emissiveTriCount) return false;

        const int       tri_id = data.emissiveTriangleIndices[local_idx];
        const Triangle& tri    = scene.triangles[tri_id];

        // Shirley's square-to-triangle mapping for uniform area sampling.
        // Triangle::sampleSurface already implements this internally. Calling it with u.y and u.z
        // directly performs the uniform mapping correctly, avoiding double-mapping and nested square roots.
        const Point3f p_e = tri.sampleSurface(Point2f(u.y, u.z));

        const Vector3f v    = p_e - ref.p;
        const float    dist = v.length();
        const float    d2   = dist * dist;
        const Vector3f wi   = v / dist;

        // Precompute triangle normal and area together; len_cross cancels out in solid-angle PDF:
        // Area = 0.5 * len, cos_e = dot(cross_p, -wi) / len
        // pdf = (prob_tri / Area) * d2 / cos_e = 2.0f * prob_tri * d2 / dot(cross_p, -wi)
        const Vector3f e1 = tri.p1 - tri.p0;
        const Vector3f e2 = tri.p2 - tri.p0;
        const Vector3f cross_p = cross(e1, e2);
        const float dot_cross_wi = dot(cross_p, -wi);

        bool is_two_sided = true;
        if (tri.mesh_id >= 0 && (uint32_t)tri.mesh_id < scene.meshCount && scene.meshes != nullptr) {
            int eid = scene.meshes[tri.mesh_id].emitterId;
            if (eid >= 0 && (uint32_t)eid < scene.emitterCount && scene.emitters != nullptr) {
                is_two_sided = scene.emitters[eid].twoSided;
            }
        }

        const float cos_e = is_two_sided ? fabsf(dot_cross_wi) : dot_cross_wi;
        if (cos_e <= 0.f || data.emitterTriangleFuncSum <= 0.f) { es.pdf = 0.f; return true; }

        es.p            = p_e;
        es.d            = wi;
        es.dist         = dist;
        es.pdf          = 2.0f * weight_val * d2 / cos_e;
        es.delta        = false;
        es.primitive_id = tri_id;
        es.mesh_id      = tri.mesh_id;

        // Evaluate emitter radiance
        SurfaceInteraction light_si;
        light_si.shape_id     = tri.mesh_id;
        light_si.material_id  = tri.material_id;
        light_si.primitive_id = tri_id;
        light_si.front_face   = (dot_cross_wi >= 0.f);
        es.Le = scene.eval_surface_emission(light_si);
        return true;
    }

    // Calculates solid-angle PDF of sampling the area light
    HD float pdf(const Scene&   scene,
                 int             emitter_primitive_id,
                 const Vector3f& wi,
                 float           dist,
                 const CDFLightSamplerData& data) const
    {
        if (!data.emissiveGlobalToIndex ||
            emitter_primitive_id >= (int)scene.triangleCount)
            return 0.f;

        const int idx = data.emissiveGlobalToIndex[emitter_primitive_id];
        if (idx < 0 || idx >= data.emissiveTriCount) return 0.f;

        const float prob_tri = data.emitterTriangleCdf[idx + 1]
                             - data.emitterTriangleCdf[idx];

        const Triangle& tri = scene.triangles[emitter_primitive_id];
        const Vector3f e1 = tri.p1 - tri.p0;
        const Vector3f e2 = tri.p2 - tri.p0;
        const Vector3f cross_p = cross(e1, e2);
        const float dot_cross_wi = dot(cross_p, -wi);

        bool is_two_sided = true;
        if (tri.mesh_id >= 0 && (uint32_t)tri.mesh_id < scene.meshCount && scene.meshes != nullptr) {
            int eid = scene.meshes[tri.mesh_id].emitterId;
            if (eid >= 0 && (uint32_t)eid < scene.emitterCount && scene.emitters != nullptr) {
                is_two_sided = scene.emitters[eid].twoSided;
            }
        }

        const float cos_e = is_two_sided ? fabsf(dot_cross_wi) : dot_cross_wi;
        if (cos_e <= 0.f) return 0.f;

        return (2.0f * prob_tri * dist * dist) / cos_e;
    }
};

FUTABA_NAMESPACE_END
