#pragma once

#include "distribution.cuh"
#include "scene.cuh"
#include "emitter.cuh"
#include "surface_interaction.cuh"
#include "emitter_sample.cuh"
#include "area.cuh"
#include "common.cuh"
#include "launch_params.h"

FUTABA_NAMESPACE_BEGIN

struct EmitterSampler {
    LightSamplerData config;

    HD explicit EmitterSampler(const LightSamplerData& c = {}) : config(c) {}

    // Sample an emitter direction from reference surface interaction
    HD bool sample_direction(const Scene&              scene,
                             const SurfaceInteraction& ref,
                             const Point3f&             u,
                             DirectionSample&           ds,
                             Color3f&                   em_weight) const
    {
        const bool have_area = (config.cdf.emissiveTriCount > 0);
        const bool have_env  = scene.envMap.isActive();

        int num_categories = (have_area ? 1 : 0) + (have_env ? 1 : 0);
        if (num_categories == 0) return false;

        const float select_prob = 1.0f / (float)num_categories;
        const int   cat_idx     = (num_categories == 1) ? 0 : ((u.x < select_prob) ? 0 : 1);
        const float remapped_ux = (num_categories == 1) ? u.x : ((cat_idx == 0) ? (u.x / select_prob) : (u.x - select_prob) / select_prob);
        const Point3f remapped_u(fminf(remapped_ux, 0.999999f), u.y, u.z);

        bool success = false;
        if (have_area && (cat_idx == 0)) {
            AreaEmitter area_emitter;
            if (area_emitter.sample(scene, ref, remapped_u, config.cdf, ds)) {
                ds.pdf *= select_prob;
                success = (ds.pdf > 0.f);
            }
        } else if (have_env) {
            const Vector3f d = scene.envMap.sampleDirection(Point2f(remapped_u.x, remapped_u.y));
            const float pdf  = scene.envMap.pdf(d);
            if (pdf > 0.f) {
                ds.d            = d;
                ds.dist         = Infinity;
                ds.pdf          = pdf * select_prob;
                ds.Le           = scene.envMap.eval(d);
                ds.delta        = false;
                ds.primitive_id = -1;
                ds.mesh_id      = -1;
                success         = true;
            }
        }

        if (success && ds.pdf > 0.f) {
            em_weight = ds.Le / ds.pdf;
            ds.weight = em_weight;
            return true;
        }

        return false;
    }

    // Backwards-compatible sample() signature
    HD bool sample(const Scene& scene, const SurfaceInteraction& ref,
                   const Point3f& u, EmitterSample& es) const 
    {
        Color3f em_weight;
        return sample_direction(scene, ref, u, es, em_weight);
    }

    // Evaluate solid-angle PDF of sampling an emitter in direction wi
    HD float pdf_direction(const Scene& scene,
                           int emitter_primitive_id,
                           const Vector3f& wi,
                           float dist = Infinity) const
    {
        const bool have_area = (config.cdf.emissiveTriCount > 0);
        const bool have_env  = scene.envMap.isActive();

        int num_categories = (have_area ? 1 : 0) + (have_env ? 1 : 0);
        if (num_categories == 0) return 0.f;

        const float select_prob = 1.0f / (float)num_categories;

        if (emitter_primitive_id < 0) {
            return have_env ? (select_prob * scene.envMap.pdf(wi)) : 0.f;
        }

        if (have_area) {
            AreaEmitter area_emitter;
            return select_prob * area_emitter.pdf(scene, emitter_primitive_id, wi, dist, config.cdf);
        }

        return 0.f;
    }

    // Backwards-compatible pdf() signature
    HD float pdf(const Scene& scene, int emitter_primitive_id,
                 const Vector3f& wi, float dist) const
    {
        return pdf_direction(scene, emitter_primitive_id, wi, dist);
    }

    // Evaluate direct emission from hit surface or environment with MIS
    HD Color3f eval_direct_emission(const Scene& scene,
                                    const SurfaceInteraction& si,
                                    bool hit,
                                    const Ray& ray,
                                    const Point3f& prev_p,
                                    float prev_bsdf_pdf,
                                    bool prev_bsdf_delta) const
    {
        if (!hit) {
            Color3f Le = scene.eval_environment(ray.d);
            if (Le.hasPositiveComponent() && scene.use_nee && !prev_bsdf_delta) {
                float emitter_pdf = pdf_direction(scene, -1, ray.d, Infinity);
                Le *= mis_weight(prev_bsdf_pdf, emitter_pdf);
            }
            return Le;
        }

        Color3f Le = scene.eval_surface_emission(si);
        if (Le.hasPositiveComponent() && scene.use_nee && !prev_bsdf_delta) {
            Vector3f offset = si.p - prev_p;
            float dist = length(offset);
            Vector3f d = offset / fmaxf(dist, 1e-8f);
            float emitter_pdf = pdf_direction(scene, si.primitive_id, d, dist);
            Le *= mis_weight(prev_bsdf_pdf, emitter_pdf);
        }
        return Le;
    }
};

// Evaluate Next Event Estimation shadow ray & MIS contribution:
// L_nee = f_local * Le * (cos_theta * mis / pdf)
HD Color3f nee_contribution(const Scene& scene,
                            const Ray& shadow_ray, float shadow_t_max, int occluder_mesh_id,
                            const EmitterSample& es,
                            const Color3f& f_local, float pdf_local, float cos_theta)
{
    if (f_local.isZero() || es.Le.isZero())
        return Color3f(0.f);

    if (scene.occluded(shadow_ray, 1e-6f, shadow_t_max, occluder_mesh_id))
        return Color3f(0.f);

    float mis = es.delta ? 1.0f : mis_weight(es.pdf, pdf_local);
    return f_local * es.Le * (cos_theta * mis / es.pdf);
}

FUTABA_NAMESPACE_END

