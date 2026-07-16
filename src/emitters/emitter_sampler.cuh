#pragma once

#include "distribution.cuh"
#include "scene.cuh"
#include "emitter.cuh"
#include "surface_interaction.cuh"
#include "emitter_sample.cuh"
#include "area.cuh"

#include "sampler.cuh"
#include "launch_params.h"

FUTABA_NAMESPACE_BEGIN

// ---------------------------------------------------------------------------
// Emitter Sampler Interface Requirements:
// 
// Any light sampler (e.g., PowerEmitterSampler)
// must implement the following compile-time static interface:
//
// 1. A sampling method:
//    HD bool sample(const Scene& scene,
//                   const SurfaceIntersection& ref,
//                   const Point3f& u,
//                   EmitterSample& es) const;
//    - Selects an emitter and samples a point/direction on it.
//    - Populates the EmitterSample struct (solid-angle PDF, radiance, etc.).
//    - Returns true if sampling succeeded, false otherwise.
//
// 2. A probability density evaluation method:
//    HD float pdf(const Scene& scene,
//                 int emitter_primitive_id,
//                 const Vector3f& wi,
//                 float dist) const;
//    - Computes the solid-angle PDF of sampling the given direction/emitter
//      from the reference shading point.
//    - If emitter_primitive_id is -1, evaluates the PDF for environment map.
// ---------------------------------------------------------------------------
struct PowerEmitterSampler {
    CDFLightSamplerData data;

    HD explicit PowerEmitterSampler(const CDFLightSamplerData& d) : data(d) {}

    HD bool sample(const Scene&              scene,
                   const SurfaceIntersection& ref,
                   const Point3f&             u,
                   EmitterSample&             es) const
    {
        const bool have_area = (data.emissiveTriCount > 0);
        const bool have_env  = scene.envMap.isActive();

        int num_categories = 0;
        if (have_area) num_categories++;
        if (have_env) num_categories++;

        if (num_categories == 0) return false;

        const float select_prob = 1.f / num_categories;
        int cat_idx = (int)(u.x * (float)num_categories);
        if (cat_idx > num_categories - 1) cat_idx = num_categories - 1;

        const float new_ux = fminf((u.x - cat_idx * select_prob) * num_categories, 0.999999f);
        const Point3f new_u(new_ux, u.y, u.z);

        int current_cat = 0;

        if (have_area) {
            if (current_cat == cat_idx) {
                AreaEmitter area_emitter;
                if (!area_emitter.sample(scene, ref, new_u, data, es)) return false;
                es.pdf *= select_prob;
                return true;
            }
            current_cat++;
        }

        if (have_env) {
            if (current_cat == cat_idx) {
                if (!sample_env(scene, new_u, es)) return false;
                es.pdf *= select_prob;
                return true;
            }
            current_cat++;
        }

        return false;
    }

    HD float pdf(const Scene&   scene,
                 int             emitter_primitive_id,
                 const Vector3f& wi,
                 float           dist) const
    {
        const bool have_area = (data.emissiveTriCount > 0);
        const bool have_env  = scene.envMap.isActive();

        int num_categories = 0;
        if (have_area) num_categories++;
        if (have_env) num_categories++;

        if (num_categories == 0) return 0.f;
        const float select_prob = 1.f / num_categories;

        if (emitter_primitive_id < 0) {
            if (have_env) {
                return select_prob * scene.envMap.pdf(wi);
            }
            return 0.f;
        }

        if (have_area) {
            AreaEmitter area_emitter;
            return select_prob * area_emitter.pdf(scene, emitter_primitive_id, wi, dist, data);
        }

        return 0.f;
    }

private:

    HD bool sample_env(const Scene& scene, const Point3f& u,
                       EmitterSample& es) const
    {
        const Vector3f d   = scene.envMap.sampleDirection(Point2f(u.x, u.y));
        const float    pdf = scene.envMap.pdf(d);
        if (pdf <= 0.f) return false;
        es.d = d; es.dist = 1e30f; es.pdf = pdf;
        es.Le = scene.envMap.eval(d); es.delta = false;
        return true;
    }
};

struct EmitterSampler {
    LightSamplerData config;

    HD explicit EmitterSampler(const LightSamplerData& c = {}) : config(c) {}

    HD bool sample(const Scene&              scene,
                   const SurfaceIntersection& ref,
                   const Point3f&             u,
                   EmitterSample&             es) const
    {
        if (config.type == LIGHT_SAMPLER_POWER) {
            return PowerEmitterSampler{config.cdf}.sample(scene, ref, u, es);
        }
        return false;
    }

    HD float pdf(const Scene&   scene,
                 int             emitter_primitive_id,
                 const Vector3f& wi,
                 float           dist) const
    {
        if (config.type == LIGHT_SAMPLER_POWER) {
            return PowerEmitterSampler{config.cdf}.pdf(scene, emitter_primitive_id, wi, dist);
        }
        return 0.f;
    }

    // Compute the direct emission from the hit surface or environment, weighted using MIS.
    HD Color3f eval_direct_emission(const Scene& scene,
                                    const SurfaceIntersection& si,
                                    bool hit,
                                    const Ray& ray,
                                    const Point3f& prev_p,
                                    float prev_bsdf_pdf,
                                    bool prev_bsdf_delta) const
    {
        Color3f Le(0.f);
        if (!hit) {
            Le = scene.eval_environment(ray.d);
            if (Le.x > 0.f || Le.y > 0.f || Le.z > 0.f) {
                float mis = 1.f;
                if (scene.use_nee && !prev_bsdf_delta) {
                    float emitter_pdf = pdf(scene, -1, ray.d, 1e30f);
                    mis = mis_weight(prev_bsdf_pdf, emitter_pdf);
                }
                Le = mis * Le;
            }
        } else {
            Le = scene.eval_surface_emission(si);
            if (Le.x > 0.f || Le.y > 0.f || Le.z > 0.f) {
                float mis = 1.f;
                if (scene.use_nee && !prev_bsdf_delta) {
                    Vector3f d = normalize(si.p - prev_p);
                    float dist = length(si.p - prev_p);
                    float emitter_pdf = pdf(scene, si.primitive_id, d, dist);
                    mis = mis_weight(prev_bsdf_pdf, emitter_pdf);
                }
                Le = mis * Le;
            }
        }
        return Le;
    }
};

// Combines an emitter sample with an already-evaluated *local scattering*
// response (BSDF value+pdf at a surface, or phase-function value+pdf inside
// a medium) into one shadow-tested, MIS-weighted direct-lighting contribution.
//
// This is deliberately the only place shadow-ray + MIS-combine logic for NEE
// lives. It takes f_local/pdf_local/cos_or_one as plain numbers rather than a
// Material, so it works identically for surfaces (Path) and media
// (VolumetricPath) -- neither the emitter sampler nor this function needs to
// know what kind of scattering produced them. cos_or_one is the local |cos
// theta| term for a surface, or 1 for an isotropic-ish medium scattering
// event where no cosine factor applies.
HD inline Color3f nee_contribution(const Scene& scene,
                                   const Ray& shadow_ray, float shadow_t_max, int occluder_mesh_id,
                                   const EmitterSample& es,
                                   const Color3f& f_local, float pdf_local, float cos_or_one)
{
    if ((f_local.x <= 0.f && f_local.y <= 0.f && f_local.z <= 0.f) ||
        (es.Le.x <= 0.f && es.Le.y <= 0.f && es.Le.z <= 0.f))
        return Color3f(0.f);

    if (scene.occluded(shadow_ray, 1e-6f, shadow_t_max, occluder_mesh_id))
        return Color3f(0.f);

    float mis = es.delta ? 1.f : mis_weight(es.pdf, pdf_local);
    return f_local * (cos_or_one / es.pdf) * es.Le * mis;
}

FUTABA_NAMESPACE_END
