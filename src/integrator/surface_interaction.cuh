#pragma once

#include "frame.cuh"
#include "ray.cuh"
#include "types.cuh"

FUTABA_NAMESPACE_BEGIN

struct SurfaceIntersection {
    // World-space geometry
    float    t;
    Point3f  p;
    Normal3f n;

    // Incoming world-space direction
    Vector3f wi;

    // Surface IDs
    Point2f  uv;
    int      shape_id;
    int      material_id;
    int      primitive_id;

    // Shading frame (built from normal at intersection)
    Frame frame;

    bool     front_face;

    HD SurfaceIntersection()
        : t(Infinity), p(0.f), n(0.f), wi(0.f), uv(0.f),
          shape_id(-1), material_id(-1), primitive_id(-1),
          frame(), front_face(true) {}

    HD bool is_valid() const { return isfinite(t); }

    HD void set_frame_from_normal(const Vector3f& normal) {
        frame.setFromNormal(normal);
    }

    HD Vector3f to_world(const Vector3f& v) const { return frame.to_world(v); }
    HD Vector3f to_local(const Vector3f& v) const { return frame.to_local(v); }

    // Spawn an offset ray to avoid self-intersection.
    HD Ray3f spawn_ray(const Vector3f& d) const {
        const float sign = (dot(d, n) >= 0.f) ? 1.f : -1.f;
        const float eps    = fmaxf(1e-4f * t, 1e-6f);
        const Vector3f off = Vector3f(n.x, n.y, n.z) * (eps * sign);
        return Ray3f(p + off, d);
    }
};

FUTABA_NAMESPACE_END
