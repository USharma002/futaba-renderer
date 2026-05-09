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
  float t;
  Point3f p;
  Normal3f n;
  bool front_face;

  // Material data (copied from Material on hit)
  Color3f albedo;
  Color3f emission;
  float ior;
  BSDFType mat_type;

  // Incoming world-space direction
  Vector3f wi;

  // Surface IDs
  Point2f uv;
  int shape_id;
  int material_id;

  // Shading frame (built from normal at intersection)
  Frame frame;

  HD SurfaceIntersection()
      : t(INFINITY), p(0.f), n(0.f), front_face(true), albedo(0.f),
        emission(0.f), ior(1.5f), mat_type(BSDF_ID_DIFFUSE), shape_id(-1),
        material_id(-1), frame() {}

  HD bool is_valid() const { return isfinite(t); }

  HD void set_frame_from_normal(const Vector3f &normal) {
    frame.setFromNormal(normal);
  }

  HD Vector3f to_world(const Vector3f &v) const { return frame.to_world(v); }
  HD Vector3f to_local(const Vector3f &v) const { return frame.to_local(v); }

  // Sets wi and front_face on bs before calling the BSDF.
  HD void prepare_bsdf(BSDFSample &bs) const {
    bs.wi = to_local(wi);
    bs.front_face = front_face;
  }

  HD Color3f sample_bsdf(BSDFSample &bs, const Point2f &s2) const {
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
  HD Ray3f spawn_ray(const Vector3f &d) const {
    float sign = (dot(d, n) >= 0.f) ? 1.f : -1.f;
    Vector3f offset = 0.001f * n * sign;
    return Ray3f(p + offset, d);
  }
};

} // namespace futaba
