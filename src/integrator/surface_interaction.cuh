#pragma once

#include "bsdf_sample.cuh"
#include "dielectric.cuh"
#include "diffuse.cuh"
#include "frame.cuh"
#include "material.cuh"
#include "mirror.cuh"
#include "ray.cuh"
#include "types.cuh"

namespace futaba {

struct SurfaceIntersection {
    // World-space geometry
    float    t;
    Point3f  p;
    Normal3f n;
    bool     front_face;

    // Material data (copied from Material on hit)
    Color3f  albedo;
    Color3f  emission;
    float    ior;
    BSDFType mat_type;

    // Incoming world-space direction
    Vector3f wi;

    // Surface IDs
    Point2f  uv;
    int      shape_id;
    int      material_id;

    // Shading frame (built from normal at intersection)
    Frame frame;

    HD SurfaceIntersection()
        : t(INFINITY), p(0.f), n(0.f), front_face(true),
          albedo(0.f), emission(0.f), ior(1.5f), mat_type(BSDF_ID_DIFFUSE),
          shape_id(-1), material_id(-1), frame() {}

    HD bool is_valid() const { return isfinite(t); }

    HD void set_frame_from_normal(const Vector3f& normal) {
        frame.setFromNormal(normal);
    }

    HD Vector3f to_world(const Vector3f& v) const { return frame.to_world(v); }
    HD Vector3f to_local(const Vector3f& v) const { return frame.to_local(v); }

    // Transforms wi into the local shading frame and stores front_face into bs
    // before the BSDF is evaluated. Must be called before any BSDF method.
    HD void prepare_bsdf(BSDFSample& bs) const {
        bs.wi         = to_local(wi);
        bs.front_face = front_face;
    }

    // Sample the BSDF at this surface point.
    //
    // Returns the importance weight  f(wi,wo) * |cos(wo)| / pdf,
    // which is identical to bs.weight after this call.
    // Use the return value (or bs.weight – they are the same object) to update
    // the path throughput. Do NOT apply both.
    HD Color3f sample_bsdf(BSDFSample& bs, const Point2f& s2) const {
        prepare_bsdf(bs);
        switch (mat_type) {
            case BSDF_ID_DIELECTRIC: {
                Dielectric bsdf(albedo, ior);
                return bsdf.sample(bs, s2);
            }
            case BSDF_ID_MIRROR: {
                Mirror bsdf(albedo);
                return bsdf.sample(bs, s2);
            }
            default: {
                Diffuse bsdf(albedo);
                return bsdf.sample(bs, s2);
            }
        }
    }

    // Spawn an offset ray to avoid self-intersection.
    //
    // The offset is proportional to the intersection distance t so that it
    // works correctly across scenes of wildly different scales (millimetres
    // to kilometres). A lower-bound floor prevents zero offset at very close
    // hits.
    HD Ray3f spawn_ray(const Vector3f& d) const {
        const float sign = (dot(d, n) >= 0.f) ? 1.f : -1.f;
        // 1e-4 * t gives ~0.01% of the ray length; clamped to ≥1e-6 absolute.
        const float eps    = fmaxf(1e-4f * t, 1e-6f);
        const Vector3f off = Vector3f(n.x, n.y, n.z) * (eps * sign);
        return Ray3f(p + off, d);
    }
};

} // namespace futaba
