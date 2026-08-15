#pragma once
#include "common.cuh"
#include "surface_interaction.cuh"
#include "bbox.cuh"
#include "warp.cuh"

#include "material.cuh"

FUTABA_NAMESPACE_BEGIN

struct Triangle {    
    // Triangle vertices and normals in world space (72 bytes)
    Point3f p0;
    Point3f p1;
    Point3f p2;

    Vector3f n0;
    Vector3f n1;
    Vector3f n2;

    // UV coordinates (24 bytes)
    Point2f uv0 = Point2f(0.f);
    Point2f uv1 = Point2f(0.f);
    Point2f uv2 = Point2f(0.f);

    // Material and shape IDs (8 bytes)
    int material_id = 0;
    int mesh_id = -1;
    
    // Flags (4 bytes aligned)
    bool has_normals = false;
    bool has_uvs = false;
    uint8_t _pad[2] = {0, 0};

    HD bool intersect(const Ray& r, float t_min, float t_max, SurfaceInteraction& rec, bool use_vertex_normals, int primitive_id = -1, const Material* mat = nullptr) const {
        Vector3f edge1 = p1 - p0;
        Vector3f edge2 = p2 - p0;
        Vector3f pvec = cross(r.d, edge2);
        float det = dot(edge1, pvec);
        if (det > -1e-8f && det < 1e-8f) return false;
        float inv_det = 1.0f / det;
        Vector3f tvec = r.o - p0;
        float u = dot(tvec, pvec) * inv_det;
        if (u < 0.0f || u > 1.0f) return false;
        Vector3f qvec = cross(tvec, edge1);
        float v = dot(r.d, qvec) * inv_det;
        if (v < 0.0f || u + v > 1.0f) return false;
        float t = dot(edge2, qvec) * inv_det;
        if (t < t_min || t > t_max) return false;
        populate_intersection(r, t, u, v, rec, use_vertex_normals, primitive_id, mat);
        return true;
    }

    HD Frame compute_tangent_frame(const Vector3f& n) const {
        if (has_uvs) {
            Vector3f e1 = p1 - p0;
            Vector3f e2 = p2 - p0;
            float du1 = uv1.x - uv0.x;
            float dv1 = uv1.y - uv0.y;
            float du2 = uv2.x - uv0.x;
            float dv2 = uv2.y - uv0.y;
            float det = du1 * dv2 - du2 * dv1;
            if (fabsf(det) > 1e-7f) {
                float inv_det = 1.0f / det;
                Vector3f dpdu = (e1 * dv2 - e2 * dv1) * inv_det;
                Vector3f s = dpdu - n * dot(n, dpdu);
                if (dot(s, s) > 1e-12f) {
                    s = normalize(s);
                    Vector3f t = cross(n, s);
                    return Frame(s, t, n);
                }
            }
        }
        return Frame(n);
    }

    HD void apply_normal_map(const Material& mat, SurfaceInteraction& rec) const {
#ifdef __CUDA_ARCH__
        if (mat.normalMapTexObj != 0) {
            Frame tangentFrame = compute_tangent_frame(rec.n);
            float4 nm = tex2D<float4>(mat.normalMapTexObj, rec.uv.x, rec.uv.y);
            Vector3f nLocal(nm.x * 2.0f - 1.0f, nm.y * 2.0f - 1.0f, nm.z * 2.0f - 1.0f);
            if (dot(nLocal, nLocal) > 0.01f) {
                rec.n = normalize(tangentFrame.to_world(normalize(nLocal)));
                rec.set_frame_from_normal(rec.front_face ? rec.n : -rec.n);
            }
        }
#endif
    }

    HD void populate_intersection(const Ray& r, float t, float u, float v, SurfaceInteraction& rec, bool use_vertex_normals, int primitive_id = -1, const Material* mat = nullptr) const {
        rec.t = t;
        rec.p = r(t);
        const float w = 1.0f - u - v;
        const Vector3f face_n = normalize(cross(p1 - p0, p2 - p0));
        rec.n = (has_normals && use_vertex_normals) ? normalize(n0 * w + n1 * u + n2 * v) : face_n;
        rec.wi = -r.d;
        rec.shape_id = mesh_id;
        rec.material_id = material_id;
        rec.primitive_id = primitive_id;
        rec.front_face = dot(r.d, face_n) < 0.0f;
        rec.uv = has_uvs ? (uv0 * w + uv1 * u + uv2 * v) : Point2f(u, v);
        rec.set_frame_from_normal(rec.front_face ? rec.n : -rec.n);

        if (mat != nullptr) {
            apply_normal_map(*mat, rec);
        }
    }

    HD Point3f sampleSurface(const Point2f& s) const {
        Point2f bary = Warp::squareToUniformTriangle(s);
        return p0 + bary.x * (p1 - p0) + bary.y * (p2 - p0);
    }

    HD AABB bounds() const {
        AABB b;
        b.expand(p0);
        b.expand(p1);
        b.expand(p2);
        return b;
    }

    HD Point3f centroid() const {
        return (p0 + p1 + p2) * (1.0f / 3.0f);
    }
};


FUTABA_NAMESPACE_END