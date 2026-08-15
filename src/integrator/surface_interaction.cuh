#pragma once

#include "frame.cuh"
#include "ray.cuh"
#include "types.cuh"

FUTABA_NAMESPACE_BEGIN

struct SurfaceInteraction {
    // Shading frame (built from normal at intersection) (36 bytes)
    Frame frame;

    // World-space geometry (24 bytes)
    Point3f  p;
    Normal3f n;

    // Incoming world-space direction (12 bytes)
    Vector3f wi;

    // Surface UVs (8 bytes)
    Point2f  uv;

    // Ray parameter and Surface IDs (16 bytes)
    float    t;
    int      shape_id;
    int      material_id;
    int      primitive_id;

    // Flags (4 bytes aligned)
    bool     front_face;
    uint8_t  _pad[3] = {0, 0, 0};

    HD SurfaceInteraction()
        : frame(), p(0.f), n(0.f), wi(0.f), uv(0.f),
          t(Infinity), shape_id(-1), material_id(-1), primitive_id(-1),
          front_face(true) {}

    HD bool is_valid() const { return isfinite(t); }

    HD void set_frame_from_normal(const Vector3f& normal) {
        frame.setFromNormal(normal);
    }

    HD Vector3f to_world(const Vector3f& v) const { return frame.to_world(v); }
    HD Vector3f to_local(const Vector3f& v) const { return frame.to_local(v); }

    // Spawn an offset ray to avoid self-intersection.
    HD Ray3f spawn_ray(const Vector3f& d) const {
        const float sign   = (dot(d, n) >= 0.f) ? 1.f : -1.f;
        const float eps    = 1e-4f;
        const Vector3f off = n * (eps * sign);
        return Ray3f(p + off, d);
    }
};

FUTABA_NAMESPACE_END
