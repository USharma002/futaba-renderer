#pragma once

#include "bsdf_sample.cuh"
#include "emitter_sampler.cuh"
#include "sampler.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"
#include "guiding_device.cuh"
#include "training_buffer.h"
#include <cmath>

namespace futaba {

HD float mis_weight(float pdf_a, float pdf_b)
{
    const float a2 = pdf_a * pdf_a;
    return (pdf_a > 0.f) ? a2 / (a2 + pdf_b * pdf_b) : 0.f;
}

struct Path {
    int             max_depth;
    int             rr_depth;
    EmitterSampler emitter_sampler;

    HD explicit Path(int max_d = 12, int rr_d = 5, EmitterSampler sampler = EmitterSampler())
        : max_depth(max_d), rr_depth(rr_d), emitter_sampler(sampler) {}

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
        return !scene.occluded(shadow, 1e-6f, t_max, es.mesh_id);
    }

    template<bool RecordDenoiseGuides>
    HD void record_denoiser_guides(const SurfaceIntersection& si, bool hit, int depth, const Ray& current_ray,
                                   Color3f* denoise_albedo_buffer, Color3f* denoise_normal_buffer,
                                   int pixel_index, int sample_count, const PerspectiveCamera* camera,
                                   const Scene& scene) const
    {
        if constexpr (RecordDenoiseGuides) {
            if (depth == 0) {
                Color3f alb(0.f);
                Vector3f norm(0.f);
                if (hit) {
                    alb = si.albedo;
                    if (camera) {
                        norm = Vector3f(dot(si.n, camera->right),
                                        dot(si.n, camera->trueUp),
                                        dot(si.n, camera->forward));
                    }
                } else {
                    alb = scene.eval_environment(current_ray.d);
                }
                if (sample_count == 1) {
                    denoise_albedo_buffer[pixel_index] = alb;
                    denoise_normal_buffer[pixel_index] = norm;
                } else {
                    denoise_albedo_buffer[pixel_index] += alb;
                    denoise_normal_buffer[pixel_index] += norm;
                }
    }
    }
    }

    // Record training features dynamically based on guiding mode.
    template<bool RecordTraining>
    HD void record_training_features(const SurfaceIntersection& si, bool hit, int depth, const Ray& current_ray,
                                     const TrainingBuffers& tb, bool record_training, int guiding_mode) const
    {
        if constexpr (RecordTraining) {
            if (record_training && depth < tb.max_depth) {
                int buf_idx = depth * tb.img_size + tb.pixel_index;
                tb.active[buf_idx] = 1.0f;

                if (hit) {
                    if (guiding_mode == PATH_GUIDING_PPG || guiding_mode == PATH_GUIDING_NPM) {
                        if (tb.position) tb.position[buf_idx] = si.p;
                    }
                    if (guiding_mode == PATH_GUIDING_NPM) {
                        if (tb.normals) tb.normals[buf_idx] = si.n;
                        if (tb.wi) tb.wi[buf_idx] = si.to_local(-current_ray.d);
                        if (tb.material_id) tb.material_id[buf_idx] = (float)si.material_id;
                    }
                } else {
                    tb.active[buf_idx] = 0.0f; // Environmental hit is inactive geometry
                    if (guiding_mode == PATH_GUIDING_NPM) {
                        if (tb.material_id) tb.material_id[buf_idx] = -1.0f;
                    }
                }
            }
        }
    }

    // Propagate radiance backward along the path.
    template<bool RecordTraining>
    HD void backward_propagate_radiance(int final_depth, const TrainingBuffers& tb, bool record_training, int guiding_mode,
                                        const Color3f* local_Le, const Color3f* local_nee, const Color3f* local_bsdf) const
    {
        if constexpr (RecordTraining) {
            if (record_training && tb.radiance && (guiding_mode == PATH_GUIDING_PPG || guiding_mode == PATH_GUIDING_NPM)) {
                Color3f incoming = Color3f(0.f);
                constexpr int MAX_LOCAL_DEPTH = 16;
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
        }
    }

    template<bool RecordTraining = false, bool RecordDenoiseGuides = false>
    HD Color3f sample(const Ray& ray, const Scene& scene, Sampler& sampler,
                      const TrainingBuffers& tb = TrainingBuffers(),
                      Color3f* denoise_albedo_buffer = nullptr,
                      Color3f* denoise_normal_buffer = nullptr,
                      int pixel_index = -1,
                      int sample_count = 1,
                      const PerspectiveCamera* camera = nullptr,
                      int guiding_mode = 0) const
    {
        Color3f L(0.f), beta(1.f);
        float   eta = 1.f;
        Ray     current_ray = ray;

        float   prev_bsdf_pdf   = 1.f;
        bool    prev_bsdf_delta = true;
        Point3f prev_p          = ray.o;

        [[maybe_unused]] bool record_training = RecordTraining && (tb.active != nullptr);

        // Pre-clear active flags when recording
        if constexpr (RecordTraining) {
            if (record_training) {
                for (int d = 0; d < tb.max_depth; ++d) {
                    int buf_idx = d * tb.img_size + tb.pixel_index;
                    tb.active[buf_idx] = 0.0f;
                }
            }
        }

        // Local register arrays for backward radiance propagation.
        // Only allocated when training is active to avoid register spill.
        constexpr int MAX_LOCAL_DEPTH = RecordTraining ? 16 : 1;
        Color3f local_Le[MAX_LOCAL_DEPTH];
        Color3f local_nee[MAX_LOCAL_DEPTH];
        Color3f local_bsdf[MAX_LOCAL_DEPTH];

        if constexpr (RecordTraining) {
            if (record_training) {
                for (int i = 0; i < MAX_LOCAL_DEPTH; ++i) {
                    local_Le[i] = Color3f(0.f);
                    local_nee[i] = Color3f(0.f);
                    local_bsdf[i] = Color3f(0.f);
                }
            }
        }

        [[maybe_unused]] int final_depth = 0;

        for (int depth = 0; depth < max_depth; ++depth) {
            if constexpr (RecordTraining) {
                final_depth = depth + 1;
            }
            SurfaceIntersection si;
            bool hit = scene.intersect(current_ray, current_ray.mint, current_ray.maxt, si);

            // Record denoiser guides at the first bounce
            record_denoiser_guides<RecordDenoiseGuides>(si, hit, depth, current_ray,
                                                       denoise_albedo_buffer, denoise_normal_buffer,
                                                       pixel_index, sample_count, camera, scene);

            // Record training features for this bounce
            record_training_features<RecordTraining>(si, hit, depth, current_ray, tb, record_training, guiding_mode);

            if (!hit) {
                // Environment emission
                const Color3f Le = scene.eval_environment(current_ray.d);
                if constexpr (RecordTraining) {
                    if (record_training && depth < MAX_LOCAL_DEPTH) {
                        local_Le[depth] = Le;
                    }
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

            if constexpr (RecordTraining) {
                if (record_training && depth < MAX_LOCAL_DEPTH) {
                    local_Le[depth] = emission;
                }
            }

            // NEE direct lighting
            Color3f nee_val(0.f);
            if (scene.use_nee && !si.is_bsdf_delta()) {
                nee_val = nee_contribution(scene, si, sampler);
                L += beta * nee_val;
            }
            if constexpr (RecordTraining) {
                if (record_training && depth < MAX_LOCAL_DEPTH) {
                    local_nee[depth] = nee_val;
                }
            }

            // BSDF sample
            BSDFSample bs;
            const Color3f bsdf_w = si.sample_bsdf(bs, sampler.next2D());
            if (!bs.is_valid()) break;

            // Record bsdf weight and wo direction
            if constexpr (RecordTraining) {
                if (record_training && depth < MAX_LOCAL_DEPTH) {
                    local_bsdf[depth] = bsdf_w * bs.pdf;
                }
                if (record_training && depth < tb.max_depth && tb.wo) {
                    if (guiding_mode == PATH_GUIDING_PPG || guiding_mode == PATH_GUIDING_NPM) {
                        int buf_idx = depth * tb.img_size + tb.pixel_index;
                        tb.wo[buf_idx] = bs.wo;
                    }
                }
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
        backward_propagate_radiance<RecordTraining>(final_depth, tb, record_training, guiding_mode, local_Le, local_nee, local_bsdf);

        return L;
    }
};

using PathIntegrator = Path;

} // namespace futaba
