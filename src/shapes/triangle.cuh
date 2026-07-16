#pragma once
#include "common.cuh"
#include "surface_interaction.cuh"
#include "bbox.cuh"
#include "warp.cuh"

FUTABA_NAMESPACE_BEGIN

struct Triangle {    
    // Material and shape IDs for intersection record
    int material_id;          // Material ID (4 bytes)
    int mesh_id = -1;         // Mesh ID (4 bytes)
    
    bool has_normals = false; // Whether the triangle has vertex normals
    bool has_uvs = false;     // Whether the triangle has vertex UV coordinates
    
    Point2f uv0 = Point2f(0.f); // UV coordinates for vertex 0
    Point2f uv1 = Point2f(0.f); // UV coordinates for vertex 1
    Point2f uv2 = Point2f(0.f); // UV coordinates for vertex 2
    
    // Triangle vertices and normals in world space
    Point3f p0;
    Point3f p1;
    Point3f p2;

    Vector3f n0;
    Vector3f n1;
    Vector3f n2;

    HD bool intersect(const Ray& r, float t_min, float t_max, SurfaceIntersection& rec, bool use_vertex_normals, int primitive_id = -1) const {
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
        populate_intersection(r, t, u, v, rec, use_vertex_normals, primitive_id);
        return true;
    }

    HD void populate_intersection(const Ray& r, float t, float u, float v, SurfaceIntersection& rec, bool use_vertex_normals, int primitive_id = -1) const {
        rec.t = t;
        rec.p = r(rec.t);
        Vector3f edge1 = p1 - p0;
        Vector3f edge2 = p2 - p0;
        Vector3f face_n = normalize(cross(edge1, edge2));
        if (has_normals && use_vertex_normals) {
            float w = 1.0f - u - v;
            rec.n = normalize(n0 * w + n1 * u + n2 * v);
        } else {
            rec.n = face_n;
        }
        rec.wi = -r.d;
        rec.shape_id = mesh_id;
        rec.material_id = material_id;
        rec.primitive_id = primitive_id;
        rec.front_face = dot(r.d, face_n) < 0.0f;
        if (has_uvs) {
            float w = 1.0f - u - v;
            rec.uv = Point2f(
                uv0.x * w + uv1.x * u + uv2.x * v,
                uv0.y * w + uv1.y * u + uv2.y * v
            );
        } else {
            rec.uv = Point2f(u, v);
        }
        Vector3f frame_n = rec.front_face ? rec.n : -rec.n;
        rec.set_frame_from_normal(frame_n);
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
        return Point3f((p0.x + p1.x + p2.x) / 3.0f,
                       (p0.y + p1.y + p2.y) / 3.0f,
                       (p0.z + p1.z + p2.z) / 3.0f);
    }
};


FUTABA_NAMESPACE_END