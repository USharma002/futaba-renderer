#pragma once

#include "path.cuh"
#include "sampler.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"
#include <cmath>

namespace futaba {

// Volumetric path tracing scaffold for future medium support.
//
// This is intentionally conservative: it preserves current behavior by falling
// back to the surface Path integrator until full medium boundary tracking,
// phase sampling, and emitter transmittance are implemented.
struct VolumetricPath {
    int  max_depth;
    int  rr_depth;
    bool hide_emitters;

    HD VolumetricPath(int max_d = 8, int rr_d = 5, bool hide_e = false)
        : max_depth(max_d), rr_depth(rr_d), hide_emitters(hide_e) {}

    HD static float mis_weight(float pdf_a, float pdf_b) {
        const float a2 = pdf_a * pdf_a;
        const float b2 = pdf_b * pdf_b;
        const float d  = a2 + b2;
        if (d <= 0.f)
            return 0.f;
        const float w = a2 / d;
        return isfinite(w) ? w : 0.f;
    }

    HD Color3f sample_incoming_radiance(const Point3f& p,
                                        const Vector3f& wi,
                                        const Scene& scene) const {
        Ray shadow(p, wi);
        SurfaceIntersection lsi;
        if (scene.intersect(shadow, shadow.mint, shadow.maxt, lsi))
            return scene.eval_surface_emission(lsi);
        return scene.eval_environment(wi);
    }

    // Minimal volumetric entry point:
    // - If medium is null, fallback to surface path tracing.
    // - If medium exists, do one free-flight sample and estimate single
    //   scattering from sampled incident direction.
    //
    // This is an intentionally simple starting point, not a full volumetric
    // path tracer with proper spectral MIS and null-collision handling.
    HD Color3f sample(const Ray& ray, const Scene& scene, Sampler& sampler) const {
        // Medium support removed for now — fall back to surface-only Path
        Path surface_path(max_depth, rr_depth);
        return surface_path.sample(ray, scene, sampler);
    }

    // ---------------------------------------------------------------------
    // Planned implementation notes
    // ---------------------------------------------------------------------
    // 1) Add medium boundary tracking (inside/outside transitions).
    // 2) Sample free-flight events against majorant and classify events.
    // 3) Implement phase sampling (isotropic + HG) and phase PDFs.
    // 4) Add transmittance-aware emitter sampling for NEE in media.
    // 5) Apply MIS between phase/BSDF sampling and emitter sampling.
    // 6) Add null-collision support for heterogeneous media.
};

} // namespace futaba
