#pragma once

#include "bsdf_sample.cuh"
#include "bsdf.cuh"
#include "emitter_sampler.cuh"
#include "sampler.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"
#include "path_recorder.cuh"
#include <cmath>

FUTABA_NAMESPACE_BEGIN

struct Path {
    int            max_depth;
    int            rr_depth;
    EmitterSampler emitter_sampler;

    HD explicit Path(int max_d = 12, int rr_d = 5, EmitterSampler sampler = EmitterSampler())
        : max_depth(max_d), rr_depth(rr_d), emitter_sampler(sampler) {}

    HD Color3f sample(const Ray& ray, const Scene& scene, Sampler& sampler,
                      const PathRecorder* recorder = nullptr) const
    {
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
            L += beta * Le;

            if (!hit)
                break;

            const Material& mat = scene.materials[si.material_id];
            const bool scatterable = !BSDF::is_delta(mat.type) && mat.type != BSDF_ID_NULL;

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

                        Color3f nee = nee_contribution(scene, si.spawn_ray(ds.d), ds.dist - 1e-4f,
                                                       ds.mesh_id, ds, f_bsdf, bsdf_pdf, cos_theta);
                        L += beta * nee;
                    }
                }
            }

            // --------------------- BSDF Sampling -------------------------
            BSDFSample bs;
            Color3f bsdf_w = BSDF::sample(mat, si, bs, sampler.next2D());
            if (!bs.is_valid())
                break;

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

        return L;
    }
};

using PathIntegrator = Path;

FUTABA_NAMESPACE_END
