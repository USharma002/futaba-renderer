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

struct TrainingBuffers {
    float* active = nullptr;
    Point3f* position = nullptr;
    Color3f* normals = nullptr;
    Color3f* wi = nullptr;
    Color3f* wo = nullptr;
    Color3f* radiance = nullptr;
    float* material_id = nullptr;
    int max_depth = 0;
    int pixel_index = -1;
    int img_size = 0;
};

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
        // Evaluate the emission at the current vertex; early exit if zero contribution
        const Color3f Le = scene.eval_surface_emission(si);
        if (Le.x <= 0.f && Le.y <= 0.f && Le.z <= 0.f) return Color3f(0.f);

        float w = 1.f;
        if (scene.use_nee && !prev_bsdf_delta) {
            // probability of getting current light in nee before
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
        // Sample a point on emitter
        EmitterSample es;
        const Point3f u3(sampler.next1D(), sampler.next1D(), sampler.next1D());
        
        if (!emitter_sampler.sample(scene, si, u3, es) || es.pdf <= 0.f)
            return Color3f(0.f);

        // Early exit if the emitter has zero radiance contribution
        if (es.Le.x <= 0.f && es.Le.y <= 0.f && es.Le.z <= 0.f)
            return Color3f(0.f);

        // direction of the shadow ray in local space of the surface interaction
        const Vector3f wo_local = si.to_local(es.d);
        const float cos_s = wo_local.z;
        if (cos_s <= 0.f) return Color3f(0.f);

        // Evaluate BSDF and its PDF for the sampled emitter direction
        Color3f f_bsdf; float pdf_bsdf;
        si.eval_pdf_bsdf(wo_local, f_bsdf, pdf_bsdf);

        // Early exit if the BSDF evaluates to zero (avoid tracing shadow ray)
        if (f_bsdf.x <= 0.f && f_bsdf.y <= 0.f && f_bsdf.z <= 0.f)
            return Color3f(0.f);

        // Shadow test (emitter self-occlusion bypass via mesh_id) - deferred until cheap checks pass
        if (!visibility_test(scene, si, es))
            return Color3f(0.f);

        // MIS weight: if the BSDF is delta, we have no choice but to take the emitter sample
        const float w = es.delta ? 1.f : mis_weight(es.pdf, pdf_bsdf);
        return f_bsdf * (cos_s / es.pdf) * es.Le * w;
    }

    HD bool visibility_test(const Scene& scene,
                        const SurfaceIntersection& si,
                        const EmitterSample& es) const
{
    const Ray   shadow = si.spawn_ray(es.d);
    const float t_max  = es.dist - 1e-4f;

    // If no self-occlusion possible, use the cheap any-hit query
    if (es.mesh_id < 0)
        return !scene.occluded(shadow, 1e-6f, t_max);

    // Otherwise, one full intersect to check self/delta in a single traversal
    SurfaceIntersection occ;
    if (!scene.intersect(shadow, 1e-6f, t_max, occ))
        return true;  // nothing blocking -> visible

    const bool is_self  = (occ.shape_id == es.mesh_id);
    const bool is_delta = (occ.mat_type == BSDF_ID_MIRROR ||
                           occ.mat_type == BSDF_ID_DIELECTRIC);
    return is_self || is_delta;
    }

    HD Color3f sample(const Ray& ray, const Scene& scene, Sampler& sampler,
                      const TrainingBuffers& tb = TrainingBuffers()) const
    {
        Color3f L(0.f), beta(1.f);
        float   eta = 1.f;
        Ray     current_ray = ray;

        float   prev_bsdf_pdf   = 1.f;
        bool    prev_bsdf_delta = true;
        Point3f prev_p          = ray.o;

        bool record_training = (tb.active && tb.pixel_index >= 0);

        // Pre-clear active flags when recording
        if (record_training) {
            for (int d = 0; d < tb.max_depth; ++d) {
                int buf_idx = d * tb.img_size + tb.pixel_index;
                tb.active[buf_idx] = 0.0f;
            }
        }

        // Local register arrays for backward radiance propagation.
        // Only allocated when training is active to avoid register spill.
        const int MAX_LOCAL_DEPTH = 32;
        Color3f local_Le[MAX_LOCAL_DEPTH];
        Color3f local_nee[MAX_LOCAL_DEPTH];
        Color3f local_bsdf[MAX_LOCAL_DEPTH];

        if (record_training) {
            for (int i = 0; i < MAX_LOCAL_DEPTH; ++i) {
                local_Le[i] = Color3f(0.f);
                local_nee[i] = Color3f(0.f);
                local_bsdf[i] = Color3f(0.f);
            }
        }

        int final_depth = 0;

        for (int depth = 0; depth < max_depth; ++depth) {
            final_depth = depth + 1;
            SurfaceIntersection si;
            bool hit = scene.intersect(current_ray, current_ray.mint, current_ray.maxt, si);

            // Record training features for this bounce (input features)
            if (record_training && depth < tb.max_depth) {
                int buf_idx = depth * tb.img_size + tb.pixel_index;
                tb.active[buf_idx] = 1.0f;

                if (hit) {
                    if (tb.position) tb.position[buf_idx] = si.p;
                    if (tb.normals) tb.normals[buf_idx] = si.n;
                    if (tb.wi) tb.wi[buf_idx] = si.to_local(-current_ray.d);
                    if (tb.material_id) tb.material_id[buf_idx] = (float)si.material_id;
                } else {
                    tb.active[buf_idx] = 0.0f; // Environmental hit is inactive geometry
                    if (tb.material_id) tb.material_id[buf_idx] = -1.0f;
                }
            }

            if (!hit) {
                // Environment emission
                const Color3f Le = scene.eval_environment(current_ray.d);
                if (record_training && depth < MAX_LOCAL_DEPTH) {
                    local_Le[depth] = Le;
                }
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
            Color3f emission = emission_weight(scene, si, prev_p,
                                               prev_bsdf_pdf, prev_bsdf_delta);
            L += beta * emission;

            if (record_training && depth < MAX_LOCAL_DEPTH) {
                local_Le[depth] = emission;
            }

            // NEE direct lighting
            Color3f nee_val(0.f);
            if (scene.use_nee && !si.is_bsdf_delta()) {
                nee_val = nee_contribution(scene, si, sampler);
                L += beta * nee_val;
            }
            if (record_training && depth < MAX_LOCAL_DEPTH) {
                local_nee[depth] = nee_val;
            }

            // BSDF sample
            BSDFSample bs;
            const Color3f bsdf_w = si.sample_bsdf(bs, sampler.next2D());
            if (!bs.is_valid()) break;

            // Record bsdf weight and wo direction
            if (record_training && depth < MAX_LOCAL_DEPTH) {
                local_bsdf[depth] = bsdf_w * bs.pdf;
            }
            if (record_training && depth < tb.max_depth && tb.wo) {
                int buf_idx = depth * tb.img_size + tb.pixel_index;
                tb.wo[buf_idx] = bs.wo;
            }

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

        // Backward propagation of radiance from local register arrays
        if (record_training && tb.radiance) {
            Color3f incoming = Color3f(0.f);
            int start_depth = (final_depth < MAX_LOCAL_DEPTH) ? final_depth - 1 : MAX_LOCAL_DEPTH - 1;
            for (int d = start_depth; d >= 0; --d) {
                if (d < tb.max_depth) {
                    int buf_idx = d * tb.img_size + tb.pixel_index;
                    if (tb.active[buf_idx] > 0.5f) {
                        incoming = local_Le[d] + local_nee[d] + local_bsdf[d] * incoming;
                        tb.radiance[buf_idx] = incoming;
                    } else {
                        incoming = Color3f(0.f);
                        tb.radiance[buf_idx] = Color3f(0.f);
                    }
                }
            }
        }

        return L;
    }
};

using PathIntegrator = Path<UniformEmitterSampler>;

} // namespace futaba
