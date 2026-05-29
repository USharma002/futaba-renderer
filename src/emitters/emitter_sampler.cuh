#pragma once

#include "distribution.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"
#include "emitter_sample.cuh"
#include "area.cuh"
#include "power_sampler.cuh"
#include "launch_params.h"

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

    HD bool sample(const Scene&              scene,
                   const SurfaceIntersection& ref,
                   const Point3f&             u,
                   EmitterSample&             es) const
    {
        const bool have_area = (scene.emissiveTriCount > 0);
        const bool have_nonarea = (scene.nonAreaEmitterCount > 0 && scene.nonAreaEmitterIndices != nullptr);
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
                if (!area_emitter.sample(scene, ref, new_u, es)) return false;
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
        const bool have_area = (scene.emissiveTriCount > 0);
        const bool have_nonarea = (scene.nonAreaEmitterCount > 0 && scene.nonAreaEmitterIndices != nullptr);
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
            return select_prob * area_emitter.pdf(scene, emitter_primitive_id, wi, dist);
        }

        return 0.f;
    }

private:

    HD bool sample_non_area(const Scene& scene,
                            const SurfaceIntersection& ref,
                            const Point3f& u,
                            EmitterSample& es) const
    {
        int sel = (int)(u.x * (float)scene.nonAreaEmitterCount);
        if (sel > scene.nonAreaEmitterCount - 1) sel = scene.nonAreaEmitterCount - 1;
        const int eid = scene.nonAreaEmitterIndices[sel];
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

    HD explicit EmitterSampler(int t = LIGHT_SAMPLER_UNIFORM) : type(t) {}

    HD bool sample(const Scene&              scene,
                   const SurfaceIntersection& ref,
                   const Point3f&             u,
                   EmitterSample&             es) const
    {
        if (type == LIGHT_SAMPLER_UNIFORM) {
            return UniformEmitterSampler{}.sample(scene, ref, u, es);
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
            return UniformEmitterSampler{}.pdf(scene, emitter_primitive_id, wi, dist);
        } else if (type == LIGHT_SAMPLER_POWER) {
            return PowerEmitterSampler{}.pdf(scene, emitter_primitive_id, wi, dist);
        }
        return 0.f;
    }
};

} // namespace futaba
