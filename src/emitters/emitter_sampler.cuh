#pragma once

#include "distribution.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"
#include "emitter_sample.cuh"
#include "area.cuh"
#include "power_sampler.cuh"
#include "launch_params.h"
#include "sampler.cuh"
#include "guiding_device.cuh"

namespace futaba {

// ---------------------------------------------------------------------------
// Emitter Sampler Interface Requirements:
// 
// Any light sampler (e.g., UniformEmitterSampler, PowerEmitterSampler)
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
struct UniformEmitterSampler {
    CDFLightSamplerData data;

    HD explicit UniformEmitterSampler(const CDFLightSamplerData& d) : data(d) {}

    HD bool sample(const Scene&              scene,
                   const SurfaceIntersection& ref,
                   const Point3f&             u,
                   EmitterSample&             es) const
    {
        const bool have_area = (data.emissiveTriCount > 0);
        const bool have_nonarea = (data.nonAreaEmitterCount > 0 && data.nonAreaEmitterIndices != nullptr);
        const bool have_env  = scene.envMap.isActive();

        int num_categories = 0;
        if (have_area) num_categories++;
        if (have_nonarea) num_categories++;
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

        if (have_nonarea) {
            if (current_cat == cat_idx) {
                if (!sample_non_area(scene, ref, new_u, es)) return false;
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
        const bool have_nonarea = (data.nonAreaEmitterCount > 0 && data.nonAreaEmitterIndices != nullptr);
        const bool have_env  = scene.envMap.isActive();

        int num_categories = 0;
        if (have_area) num_categories++;
        if (have_nonarea) num_categories++;
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

    HD bool sample_non_area(const Scene& scene,
                            const SurfaceIntersection& ref,
                            const Point3f& u,
                            EmitterSample& es) const
    {
        int sel = (int)(u.x * (float)data.nonAreaEmitterCount);
        if (sel > data.nonAreaEmitterCount - 1) sel = data.nonAreaEmitterCount - 1;
        const int eid = data.nonAreaEmitterIndices[sel];
        if (eid < 0 || (uint32_t)eid >= scene.emitterCount) return false;

        const EmitterGPU& em = scene.emitters[eid];

        if (em.type == kEmitterTypePoint) {
            const Vector3f v  = em.position - ref.p;
            const float    d  = v.length();
            if (d < 1e-8f) return false;
            const float    d2 = d * d;
            es.p = em.position; es.d = v / d; es.dist = d;
            es.pdf = 1.f; es.Le = em.radiance / d2; es.delta = true;
            return true;
        }

        if (em.type == kEmitterTypeDirectional) {
            es.d = -em.direction; es.dist = 1e30f;
            es.pdf = 1.f; es.Le = em.radiance; es.delta = true;
            return true;
        }

        return false;
    }

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
    int type;
    CDFLightSamplerData cdf_data;

    HD explicit EmitterSampler(int t = LIGHT_SAMPLER_UNIFORM, CDFLightSamplerData d = {}) : type(t), cdf_data(d) {}

    HD bool sample(const Scene&              scene,
                   const SurfaceIntersection& ref,
                   const Point3f&             u,
                   EmitterSample&             es) const
    {
        if (type == LIGHT_SAMPLER_UNIFORM) {
            return UniformEmitterSampler{cdf_data}.sample(scene, ref, u, es);
        } else if (type == LIGHT_SAMPLER_POWER) {
            return PowerEmitterSampler{}.sample(scene, ref, u, es);
        }
        return false;
    }

    HD float pdf(const Scene&   scene,
                 int             emitter_primitive_id,
                 const Vector3f& wi,
                 float           dist) const
    {
        if (type == LIGHT_SAMPLER_UNIFORM) {
            return UniformEmitterSampler{cdf_data}.pdf(scene, emitter_primitive_id, wi, dist);
        } else if (type == LIGHT_SAMPLER_POWER) {
            return PowerEmitterSampler{}.pdf(scene, emitter_primitive_id, wi, dist);
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

    // Sample direct lighting from emitters using Next Event Estimation (NEE).
    HD Color3f sample_direct_emitter(const Scene& scene,
                                     const SurfaceIntersection& si,
                                     const Material& mat,
                                     Sampler& sampler,
                                     int guiding_mode,
                                     STreeNode* sTreeNodes,
                                     const AABB& sTreeAABB,
                                     float bsdf_sampling_fraction) const
    {
        if (!scene.use_nee || BSDF::is_delta(mat.type) || mat.type == BSDF_ID_NULL)
            return Color3f(0.f);

        // 1. Sample direction towards emitter
        EmitterSample es;
        Point3f u3(sampler.next1D(), sampler.next1D(), sampler.next1D());
        if (!sample(scene, si, u3, es) || es.pdf <= 0.f)
            return Color3f(0.f);

        if (es.Le.x <= 0.f && es.Le.y <= 0.f && es.Le.z <= 0.f)
            return Color3f(0.f);

        Vector3f wo_local = si.to_local(es.d);
        float cos_theta = wo_local.z;
        if (cos_theta <= 0.f)
            return Color3f(0.f);

        // 2. Evaluate BSDF and pdf for the sampled direction
        Color3f f_bsdf;
        float bsdf_pdf = 0.f;
        BSDF::eval_pdf(mat, si, wo_local, f_bsdf, bsdf_pdf);
        if (f_bsdf.x <= 0.f && f_bsdf.y <= 0.f && f_bsdf.z <= 0.f)
            return Color3f(0.f);

        // 3. Account for path guiding in MIS
        GuidingDistribution guiding(guiding_mode, si, sTreeNodes, sTreeAABB);
        float mixed_pdf = guiding.mixed_pdf(wo_local, bsdf_pdf, bsdf_sampling_fraction);

        // 4. Trace shadow ray to check visibility
        Ray shadow = si.spawn_ray(es.d);
        if (scene.occluded(shadow, 1e-6f, es.dist - 1e-4f, es.mesh_id))
            return Color3f(0.f);

        // 5. Compute MIS weight and direct lighting contribution
        float mis = es.delta ? 1.f : mis_weight(es.pdf, mixed_pdf);
        return f_bsdf * (cos_theta / es.pdf) * es.Le * mis;
    }
};

} // namespace futaba
