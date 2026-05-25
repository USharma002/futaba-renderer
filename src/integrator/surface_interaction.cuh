#pragma once

#include "bsdf_sample.cuh"
#include "dielectric.cuh"
#include "diffuse.cuh"
#include "frame.cuh"
#include "material.cuh"
#include "microfacet.cuh"
#include "mirror.cuh"
#include "ray.cuh"
#include "types.cuh"

namespace futaba {

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
    BSDFType mat_type;

    // Material data (copied from Material on hit)
    float    ext_ior;
    float    ior;
    float    alpha;
    Color3f  albedo;
    Color3f  specular;
    Color3f  emission;
    Color3f  conductor_eta;
    Color3f  conductor_k;

    // Shading frame (built from normal at intersection)
    Frame frame;

    bool     front_face;
    bool     is_conductor;

    HD SurfaceIntersection()
        : t(Infinity), p(0.f), n(0.f), wi(0.f), uv(0.f),
          shape_id(-1), material_id(-1), primitive_id(-1), mat_type(BSDF_ID_DIFFUSE),
          ext_ior(1.000277f), ior(1.5f), alpha(1.f),
          albedo(0.f), specular(1.f), emission(0.f), conductor_eta(0.f), conductor_k(1.f),
          frame(), front_face(true), is_conductor(false) {}

    HD bool is_valid() const { return isfinite(t); }

    HD void set_frame_from_normal(const Vector3f& normal) {
        frame.setFromNormal(normal);
    }

    HD Vector3f to_world(const Vector3f& v) const { return frame.to_world(v); }
    HD Vector3f to_local(const Vector3f& v) const { return frame.to_local(v); }

    // -----------------------------------------------------------------------
    // is_bsdf_delta()
    // Returns true only for true Dirac-delta BSDFs: perfect mirror and smooth
    // dielectric.  RoughDielectric has a continuous GGX lobe and must NOT be
    // flagged as delta — doing so skips NEE at rough-glass surfaces entirely
    // and assigns MIS weight 1 to post-glass hits that should be shared.
    // -----------------------------------------------------------------------
    HD bool is_bsdf_delta() const {
        return mat_type == BSDF_ID_MIRROR || mat_type == BSDF_ID_DIELECTRIC;
    }

    // Transforms wi into the local shading frame and stores front_face into bs
    // before the BSDF is evaluated. Must be called before any BSDF method.
    HD void prepare_bsdf(BSDFSample& bs) const {
        bs.wi         = to_local(wi);
        bs.front_face = front_face;
    }

    // Sample the BSDF at this surface point.
    HD Color3f sample_bsdf(BSDFSample& bs, const Point2f& s2) const {
        prepare_bsdf(bs);
        switch (mat_type) {
            case BSDF_ID_MICROFACET: {
                Microfacet bsdf(albedo, alpha, ext_ior, ior,
                                is_conductor, conductor_eta, conductor_k, specular);
                return bsdf.sample(bs, s2);
            }
            case BSDF_ID_DIELECTRIC: {
                Dielectric bsdf(albedo, ior);
                return bsdf.sample(bs, s2);
            }
            case BSDF_ID_MIRROR: {
                Mirror bsdf(albedo);
                return bsdf.sample(bs, s2);
            }
            case BSDF_ID_NULL: {
                bs.wo = -bs.wi;
                bs.sampled_type = BSDF_ID_NULL;
                bs.pdf = 1.0f;
                bs.eta = 1.0f;
                return Color3f(1.0f);
            }
            default: {
                Diffuse bsdf(albedo);
                return bsdf.sample(bs, s2);
            }
        }
    }

    // -----------------------------------------------------------------------
    // eval_pdf_bsdf()
    //
    // Previously this called eval_bsdf_local() and pdf_bsdf_local() separately,
    // each doing a full BSDF construction + switch dispatch.  For Microfacet
    // this recomputed the half-vector, Fresnel and GGX terms twice per NEE call.
    //
    // Now we dispatch once and compute both f and pdf in a single BSDF object,
    // sharing all intermediate values.
    // -----------------------------------------------------------------------
    HD void eval_pdf_bsdf(const Vector3f& wo_local,
                           Color3f&        f_out,
                           float&          pdf_out) const
    {
        BSDFSample bs;
        bs.wi         = to_local(wi);
        bs.front_face = front_face;
        bs.wo         = wo_local;

        switch (mat_type) {
            case BSDF_ID_MICROFACET: {
                Microfacet bsdf(albedo, alpha, ext_ior, ior,
                                is_conductor, conductor_eta, conductor_k, specular);
                f_out   = bsdf.eval(bs);
                pdf_out = bsdf.pdf(bs);
                return;
            }
            case BSDF_ID_DIELECTRIC: {
                Dielectric bsdf(albedo, ior);
                f_out   = bsdf.eval(bs);
                pdf_out = bsdf.pdf(bs);
                return;
            }
            case BSDF_ID_MIRROR: {
                Mirror bsdf(albedo);
                f_out   = bsdf.eval(bs);
                pdf_out = bsdf.pdf(bs);
                return;
            }
            case BSDF_ID_NULL: {
                f_out = Color3f(0.f);
                pdf_out = 0.f;
                return;
            }
            default: {
                Diffuse bsdf(albedo);
                f_out   = bsdf.eval(bs);
                pdf_out = bsdf.pdf(bs);
                return;
            }
        }
    }

    // Spawn an offset ray to avoid self-intersection.
    HD Ray3f spawn_ray(const Vector3f& d) const {
        const float sign = (dot(d, n) >= 0.f) ? 1.f : -1.f;
        const float eps    = fmaxf(1e-4f * t, 1e-6f);
        const Vector3f off = Vector3f(n.x, n.y, n.z) * (eps * sign);
        return Ray3f(p + off, d);
    }
};

} // namespace futaba
