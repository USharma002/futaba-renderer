#pragma once

#include "bsdf_sample.cuh"
#include "emitter_sampler.cuh"
#include "sampler.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"
#include <cmath>

namespace futaba {

HD float mis_weight(float pdf_a, float pdf_b)
{
    const float a2 = pdf_a * pdf_a;
    return (pdf_a > 0.f) ? a2 / (a2 + pdf_b * pdf_b) : 0.f;
}

template<typename EmitterSamplerT = UniformEmitterSampler>
struct Path {
    int             max_depth;
    int             rr_depth;
    EmitterSamplerT emitter_sampler;

    HD explicit Path(int max_d = 12, int rr_d = 5)
        : max_depth(max_d), rr_depth(rr_d) {}

    // Compute MIS-weighted emission for a BSDF-sampled emitter hit.
    HD Color3f emission_weight(const Scene& scene,
                               const SurfaceIntersection& si,
                               const Point3f& prev_p,
                               float prev_bsdf_pdf,
                               bool  prev_bsdf_delta) const
    {
        const Color3f Le = scene.eval_surface_emission(si);
        if (Le.x <= 0.f && Le.y <= 0.f && Le.z <= 0.f) return Color3f(0.f);

        float w = 1.f;
        if (scene.use_nee && !prev_bsdf_delta) {
            const Vector3f d    = normalize(si.p - prev_p);
            const float    dist = length(si.p - prev_p);
            w = mis_weight(prev_bsdf_pdf,
                           emitter_sampler.pdf(scene, si.primitive_id, d, dist));
        }
        return w * Le;
    }

    // Evaluate NEE direct-lighting contribution at a smooth surface vertex.
    HD Color3f nee_contribution(const Scene& scene,
                                const SurfaceIntersection& si,
                                Sampler& sampler) const
    {
        EmitterSample es;
        const Point3f u3(sampler.next1D(), sampler.next1D(), sampler.next1D());
        if (!emitter_sampler.sample(scene, si, u3, es) || es.pdf <= 0.f)
            return Color3f(0.f);

        // Early exit if the emitter has zero radiance contribution
        if (es.Le.x <= 0.f && es.Le.y <= 0.f && es.Le.z <= 0.f)
            return Color3f(0.f);

        const Vector3f wo_local = si.to_local(es.d);
        const float cos_s = wo_local.z;
        if (cos_s <= 0.f) return Color3f(0.f);

        Color3f f_bsdf; float pdf_bsdf;
        si.eval_pdf_bsdf(wo_local, f_bsdf, pdf_bsdf);

        // Early exit if the BSDF evaluates to zero (avoid tracing shadow ray)
        if (f_bsdf.x <= 0.f && f_bsdf.y <= 0.f && f_bsdf.z <= 0.f)
            return Color3f(0.f);

        // Shadow test (emitter self-occlusion bypass via mesh_id) - deferred until cheap checks pass
        if (!visibility_test(scene, si, es))
            return Color3f(0.f);

        const float w = es.delta ? 1.f : mis_weight(es.pdf, pdf_bsdf);
        return f_bsdf * (cos_s / es.pdf) * es.Le * w;
    }

    // Shadow ray with emitter self-occlusion handling.
    HD bool visibility_test(const Scene& scene,
                            const SurfaceIntersection& si,
                            const EmitterSample& es) const
    {
        const Ray shadow = si.spawn_ray(es.d);
        const float t_max = es.dist - 1e-4f;

        // Fast path: use dedicated occlusion query
        if (!scene.occluded(shadow, 1e-6f, t_max))
            return true; // visible

        // Slow path: check if blocker is same-mesh emitter (self-occlusion)
        // or a delta surface (glass) using full intersection
        if (es.mesh_id < 0) return false;

        SurfaceIntersection occ;
        if (!scene.intersect(shadow, 1e-6f, t_max, occ))
            return true;

        const bool is_self = (occ.shape_id == es.mesh_id);
        const bool is_delta = (occ.mat_type == BSDF_ID_MIRROR ||
                               occ.mat_type == BSDF_ID_DIELECTRIC);
        return is_self || is_delta;
    }

    HD Color3f sample(const Ray& ray, const Scene& scene, Sampler& sampler) const
    {
        Color3f L(0.f), beta(1.f);
        float   eta = 1.f;
        Ray     current_ray = ray;

        float   prev_bsdf_pdf   = 1.f;
        bool    prev_bsdf_delta = true;
        Point3f prev_p          = ray.o;

        for (int depth = 0; depth < max_depth; ++depth) {
            SurfaceIntersection si;
            if (!scene.intersect(current_ray, current_ray.mint,
                                 current_ray.maxt, si)) {
                // Environment emission
                const Color3f Le = scene.eval_environment(current_ray.d);
                if (Le.x > 0.f || Le.y > 0.f || Le.z > 0.f) {
                    float w = 1.f;
                    if (scene.use_nee && !prev_bsdf_delta) {
                        w = mis_weight(prev_bsdf_pdf,
                                       emitter_sampler.pdf(scene, -1,
                                                           current_ray.d, 1e30f));
                    }
                    L += beta * w * Le;
                }
                break;
            }

            // Emission at current vertex (MIS against light PDF)
            L += beta * emission_weight(scene, si, prev_p,
                                        prev_bsdf_pdf, prev_bsdf_delta);

            // NEE direct lighting
            if (scene.use_nee && !si.is_bsdf_delta())
                L += beta * nee_contribution(scene, si, sampler);

            // BSDF sample
            BSDFSample bs;
            const Color3f bsdf_w = si.sample_bsdf(bs, sampler.next2D());
            if (!bs.is_valid()) break;

            beta *= bsdf_w;
            eta  *= bs.eta;

            const float max_beta = fmaxf(beta.x, fmaxf(beta.y, beta.z));
            if (max_beta <= 0.f) break;

            // Russian roulette
            if (depth >= rr_depth) {
                const float rr_prob = fminf(max_beta * eta * eta, 0.95f);
                if (sampler.next1D() >= rr_prob) break;
                beta = beta / rr_prob;
            }

            // Advance
            prev_p          = si.p;
            prev_bsdf_pdf   = bs.pdf;
            prev_bsdf_delta = si.is_bsdf_delta();
            current_ray     = si.spawn_ray(si.to_world(bs.wo));
        }

        return L;
    }
};

using PathIntegrator = Path<UniformEmitterSampler>;

} // namespace futaba
