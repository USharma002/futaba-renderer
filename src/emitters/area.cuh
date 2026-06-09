#pragma once

#include "types.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"
#include "triangle.cuh"
#include "distribution.cuh"
#include "emitter_sample.cuh"

namespace futaba {

struct AreaEmitter {
    // Samples a point on the area lights in the scene
    HD bool sample(const Scene&              scene,
                   const SurfaceIntersection& ref,
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

        // Precompute triangle normal and area together to save redundant cross products & length calculations
        const Vector3f e1 = tri.p1 - tri.p0;
        const Vector3f e2 = tri.p2 - tri.p0;
        const Vector3f cross_p = cross(e1, e2);
        const float len_cross = length(cross_p);
        if (len_cross <= 0.f || data.emitterTriangleFuncSum <= 0.f) return false;

        const float area = 0.5f * len_cross;
        const float prob_tri = weight_val;
        const float pdf_A    = prob_tri / area;
        if (pdf_A <= 0.f) return false;

        const Vector3f n_e   = cross_p / len_cross; // normalize using the precomputed length
        const float    cos_e = dot(n_e, -wi);
        if (cos_e <= 0.f) { es.pdf = 0.f; return true; }

        es.p            = p_e;
        es.d            = wi;
        es.dist         = dist;
        es.pdf          = pdf_A * d2 / cos_e;
        es.delta        = false;
        es.primitive_id = tri_id;
        es.mesh_id      = tri.mesh_id;

        // Evaluate emitter radiance
        SurfaceIntersection light_si;
        light_si.shape_id     = tri.mesh_id;
        light_si.material_id  = tri.material_id;
        light_si.primitive_id = tri_id;
        light_si.front_face   = true;
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
        const float len_cross = length(cross_p);
        if (len_cross <= 0.f) return 0.f;

        const float area = 0.5f * len_cross;
        const Vector3f n_e = cross_p / len_cross;
        const float cos_e  = dot(n_e, -wi);
        if (cos_e <= 0.f) return 0.f;

        return (prob_tri / area) * (dist * dist) / cos_e;
    }
};

} // namespace futaba
