#pragma once

#include "distribution.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"
#include "emitter_sample.cuh"
#include "area.cuh"

namespace futaba {

// Power-weighted area-light sampler with env-map fallback.
struct UniformEmitterSampler {

    HD bool sample(const Scene&              scene,
                   const SurfaceIntersection& ref,
                   const Point3f&             u,
                   EmitterSample&             es) const
    {
        const bool have_area = (scene.emissiveTriCount > 0);
        const bool have_env  = scene.envMap.isActive();

        if (!have_area && !have_env) return false;

        // Area light sampling (priority over env map when both exist)
        if (have_area) {
            AreaEmitter area_emitter;
            return area_emitter.sample(scene, ref, u, es);
        }

        // Non-area emitters (point / directional) when no area lights
        if (scene.nonAreaEmitterCount > 0 && scene.nonAreaEmitterIndices)
            return sample_non_area(scene, ref, u, es);

        // Env-map fallback
        return sample_env(scene, u, es);
    }

    HD float pdf(const Scene&   scene,
                 int             emitter_primitive_id,
                 const Vector3f& wi,
                 float           dist) const
    {
        if (emitter_primitive_id < 0) {
            if (scene.emissiveTriCount > 0) return 0.f;
            return scene.envMap.isActive() ? scene.envMap.pdf(wi) : 0.f;
        }

        AreaEmitter area_emitter;
        return area_emitter.pdf(scene, emitter_primitive_id, wi, dist);
    }

private:

    HD bool sample_non_area(const Scene& scene,
                            const SurfaceIntersection& ref,
                            const Point3f& u,
                            EmitterSample& es) const
    {
        const int sel = std::min((int)(u.x * (float)scene.nonAreaEmitterCount),
                            scene.nonAreaEmitterCount - 1);
        const int eid = scene.nonAreaEmitterIndices[sel];
        if (eid < 0 || (uint32_t)eid >= scene.emitterCount) return false;

        const EmitterGPU& em = scene.emitters[eid];

        if (em.type == kEmitterTypePoint) {
            const Vector3f v  = em.position - ref.p;
            const float    d2 = dot(v, v);
            const float    d  = sqrtf(d2);
            if (d < 1e-8f) return false;
            es.p = em.position; es.d = v * (1.f / d); es.dist = d;
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

} // namespace futaba
