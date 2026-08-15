#pragma once

#include "bsdf_sample.cuh"
#include "bsdf.cuh"
#include "emitter_sampler.cuh"
#include "guiding_device.cuh"
#include "guiding_params.h"
#include "sampler.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"
#include "path_recorder.cuh"
#include <cmath>
#include <type_traits>

FUTABA_NAMESPACE_BEGIN

constexpr int GUIDING_MAX_DEPTH = 16;

struct TrainVertex {
    Point3f  pos;
    Vector3f dir;
    Color3f  thp;
    Color3f  accum_L;
    float    pdf;
    bool     recordable;
};

struct TrainScratch {
    TrainVertex v[GUIDING_MAX_DEPTH];
    int         count = 0;
};

struct NoScratch {};

struct GuidedPath {
    int            max_depth;
    int            rr_depth;
    EmitterSampler emitter_sampler;

    HD explicit GuidedPath(int max_d = 12, int rr_d = 5, EmitterSampler sampler = EmitterSampler())
        : max_depth(max_d), rr_depth(rr_d), emitter_sampler(sampler) {}

    HD Color3f sample(const Ray& ray, const Scene& scene, Sampler& sampler,
                      const PathRecorder* recorder = nullptr,
                      const GuidingParams& guiding = GuidingParams(),
                      int pixel_index = -1) const
    {
        const bool capturing = (guiding.train_active != nullptr && pixel_index >= 0);
        const bool guided    = (guiding.active && guiding.type != GuidingType::None);

        if (capturing) {
            if (guided)
                return sample_impl<true, true>(ray, scene, sampler, recorder, guiding, pixel_index);
            else
                return sample_impl<true, false>(ray, scene, sampler, recorder, guiding, pixel_index);
        } else {
            if (guided)
                return sample_impl<false, true>(ray, scene, sampler, recorder, guiding, pixel_index);
            else
                return sample_impl<false, false>(ray, scene, sampler, recorder, guiding, pixel_index);
        }
    }

    template <bool Capture, bool Guided>
    HD Color3f sample_impl(const Ray& ray, const Scene& scene, Sampler& sampler,
                           const PathRecorder* recorder,
                           const GuidingParams& guiding,
                           int pixel_index) const
    {
        [[maybe_unused]] typename std::conditional<Capture, TrainScratch, NoScratch>::type scratch;

        Ray     current_ray = ray;
        Color3f L(0.f), beta(1.f);
        float   eta = 1.f;

        Point3f prev_p          = ray.o;
        float   prev_bsdf_pdf   = 1.f;
        bool    prev_bsdf_delta = true;

        for (int depth = 0; depth < max_depth; ++depth) {
            SurfaceInteraction si;
            bool hit = scene.intersect(current_ray, current_ray.mint, current_ray.maxt, si);

            if (recorder)
                recorder->record(depth, hit, si, current_ray, scene);

            // ------------------ Direct Emission with MIS ------------------
            Color3f Le = emitter_sampler.eval_direct_emission(scene, si, hit, current_ray, prev_p, prev_bsdf_pdf, prev_bsdf_delta);
            Color3f path_Le = beta * Le;
            L += path_Le;

            if constexpr (Capture) {
                for (int k = 0; k < scratch.count; ++k) {
                    scratch.v[k].accum_L += path_Le;
                }
            }

            if (!hit)
                break;

            const Material& mat = scene.materials[si.material_id];
            const bool scatterable = !BSDF::is_delta(mat.type) && mat.type != BSDF_ID_NULL;

            [[maybe_unused]] GuidingDistribution guide;
            if constexpr (Guided) {
                if (scatterable)
                    guide = Guiding::get_distribution(si, guiding);
            }

            // ------------------ Emitter Sampling (NEE) -------------------
            if (scene.use_nee && scatterable) {
                DirectionSample ds;
                Color3f em_weight;
                Point3f u3(sampler.next1D(), sampler.next1D(), sampler.next1D());

                if (emitter_sampler.sample_direction(scene, si, u3, ds, em_weight)) {
                    Vector3f wo_local = si.to_local(ds.d);
                    float cos_theta = Frame::cos_theta(wo_local);

                    if (cos_theta > 0.f) {
                        Color3f f_bsdf;
                        float bsdf_pdf = 0.f;
                        BSDF::eval_pdf(mat, si, wo_local, f_bsdf, bsdf_pdf);

                        float pdf_local = bsdf_pdf;
                        if constexpr (Guided) {
                            if (guide.is_valid())
                                pdf_local = guide.mixed_pdf(wo_local, bsdf_pdf, guiding.bsdf_sampling_fraction);
                        }

                        Color3f nee = nee_contribution(scene, si.spawn_ray(ds.d), ds.dist - 1e-4f,
                                                       ds.mesh_id, ds, f_bsdf, pdf_local, cos_theta);
                        Color3f path_nee = beta * nee;
                        L += path_nee;

                        if constexpr (Capture) {
                            for (int k = 0; k < scratch.count; ++k) {
                                scratch.v[k].accum_L += path_nee;
                            }
                        }
                    }
                }
            }

            // --------------------- BSDF Sampling -------------------------
            BSDFSample bs;
            Color3f bsdf_w;
            if constexpr (Guided) {
                bsdf_w = guide.is_valid()
                    ? Guiding::sample_guided(guide, mat, si, bs, sampler, guiding.bsdf_sampling_fraction)
                    : BSDF::sample(mat, si, bs, sampler.next2D());
            } else {
                bsdf_w = BSDF::sample(mat, si, bs, sampler.next2D());
            }

            if (!bs.is_valid())
                break;

            if constexpr (Capture) {
                if (scratch.count < GUIDING_MAX_DEPTH) {
                    TrainVertex& tv = scratch.v[scratch.count++];
                    tv.pos        = si.p;
                    tv.dir        = si.to_world(bs.wo);
                    tv.thp        = beta;
                    tv.accum_L    = Color3f(0.f);
                    tv.pdf        = bs.pdf;
                    tv.recordable = scatterable;
                }
            }

            // ---- Update loop variables based on current interaction ----
            beta *= bsdf_w;
            eta  *= bs.eta;

            // ------------------- Stopping Criterion --------------------
            const float max_beta = beta.maxComponent();
            if (max_beta <= 0.f) break;

            if (depth >= rr_depth) {
                const float rr_prob = fminf(fmaxf(max_beta / (eta * eta), 0.05f), 0.95f);
                if (sampler.next1D() >= rr_prob) break;
                beta /= rr_prob;
            }

            prev_p          = si.p;
            prev_bsdf_pdf   = bs.pdf;
            prev_bsdf_delta = BSDF::is_delta(mat.type);
            current_ray     = si.spawn_ray(si.to_world(bs.wo));
        }

        // ----------------- Guiding Training Output Population ------------------
        if constexpr (Capture) {
            for (int d = 0; d < scratch.count; ++d) {
                const TrainVertex& tv = scratch.v[d];
                if (tv.recordable && d < guiding.train_max_depth) {
                    float thp_scalar = tv.thp.maxComponent();
                    Color3f incident = (thp_scalar > 1e-5f) ? (tv.accum_L / thp_scalar) : Color3f(0.f);

                    const int slot = pixel_index * guiding.train_max_depth + d;
                    guiding.train_active[slot]   = 1.f;
                    guiding.train_position[slot] = tv.pos;
                    guiding.train_wo[slot]       = tv.dir;
                    guiding.train_radiance[slot] = incident;
                    guiding.train_pdf[slot]      = tv.pdf;
                }
            }
        }

        return L;
    }
};

using GuidedPathIntegrator = GuidedPath;

FUTABA_NAMESPACE_END
