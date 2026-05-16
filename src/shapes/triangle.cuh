#pragma once

#include "common.cuh"
#include "ray.cuh"
#include "diffuse.cuh"
#include "surface_interaction.cuh"

namespace futaba {

struct Triangle {
    // Triangle vertices and normals in world space
    Point3f p0;
    Point3f p1;
    Point3f p2;

    Vector3f n0;
    Vector3f n1;
    Vector3f n2;

    // Material and shape IDs for intersection record
    int material_id;
    int mesh_id = -1;
    bool has_normals = false;


    HD bool intersect(const Ray& r, float t_min, float t_max, SurfaceIntersection& rec, bool use_vertex_normals, int primitive_id = -1) const {
        // Möller–Trumbore ray-triangle intersection algorithm
        Vector3f edge1 = p1 - p0;
        Vector3f edge2 = p2 - p0;

        Vector3f pvec = cross(r.d, edge2);

        // If the determinant is near zero, the ray lies in the plane of the triangle
        float det = dot(edge1, pvec);
        if (det > -1e-8f && det < 1e-8f) return false;

        float inv_det = 1.0f / det;
        Vector3f tvec = r.o - p0;

        // If u is outside the triangle, return false
        float u = dot(tvec, pvec) * inv_det;
        if (u < 0.0f || u > 1.0f) return false;

        // If v is outside the triangle, return false
        Vector3f qvec = cross(tvec, edge1);
        float v = dot(r.d, qvec) * inv_det;
        if (v < 0.0f || u + v > 1.0f) return false;

        // Compute t to find out where the intersection point is on the line
        float t = dot(edge2, qvec) * inv_det;
        if (t < t_min || t > t_max) return false;

        rec.t = t;
        rec.p = r(rec.t);
        
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
        rec.uv = Point2f(u, v);
        Vector3f frame_n = rec.front_face ? rec.n : -rec.n;
        rec.set_frame_from_normal(frame_n);

        return true;
    }

    HD float area() const {
        return 0.5f * length(cross(p1 - p0, p2 - p0));
    }

    HD Point3f sampleSurface(const Point2f& s) const {
        float sqrt_u = sqrtf(s.x);
        float u = 1.0f - sqrt_u;
        float v = s.y * sqrt_u;
        return p0 + u * (p1 - p0) + v * (p2 - p0);
    }

    HD float pdfSurface() const {
        return 1.0f / area();
    }

};

} // namespace futaba