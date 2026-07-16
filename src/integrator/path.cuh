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
    int             max_depth;
    int             rr_depth;
    EmitterSampler emitter_sampler;

    HD explicit Path(int max_d = 12, int rr_d = 5, EmitterSampler sampler = EmitterSampler())
        : max_depth(max_d), rr_depth(rr_d), emitter_sampler(sampler) {}

    // recorder is optional: pass nullptr and nothing extra is ever touched.
    HD Color3f sample(const Ray& ray, const Scene& scene, Sampler& sampler,
                      const PathRecorder* recorder = nullptr) const
    {
        Color3f L(0.f), beta(1.f);
        float   eta = 1.f;
        Ray     current_ray = ray;

        float   prev_bsdf_pdf   = 1.f;
        bool    prev_bsdf_delta = true;
        Point3f prev_p          = ray.o;

        for (int depth = 0; depth < max_depth; ++depth) {
            SurfaceIntersection si;
            bool hit = scene.intersect(current_ray, current_ray.mint, current_ray.maxt, si);

            if (recorder)
                recorder->record(depth, hit, si, current_ray, scene);

            // ---------------------- Direct emission ----------------------

            Color3f Le = emitter_sampler.eval_direct_emission(scene, si, hit, current_ray, prev_p, prev_bsdf_pdf, prev_bsdf_delta);
            L += beta * Le;

            if (!hit)
                break;

            const Material& mat = scene.materials[si.material_id];

            // ---------------------- Emitter sampling ----------------------
            // Mirrors the reference structure: sample a light (material-free),
            // then evaluate what THIS surface's BSDF does with that direction,
            // then combine. The emitter sampler never sees `mat` -- see
            // nee_contribution() in emitter_sampler.cuh.

            if (scene.use_nee && !BSDF::is_delta(mat.type) && mat.type != BSDF_ID_NULL) {
                EmitterSample es;
                Point3f u3(sampler.next1D(), sampler.next1D(), sampler.next1D());
                if (emitter_sampler.sample(scene, si, u3, es) && es.pdf > 0.f) {
                    Vector3f wo_local = si.to_local(es.d);
                    float cos_theta = wo_local.z;
                    if (cos_theta > 0.f) {
                        Color3f f_bsdf;
                        float bsdf_pdf = 0.f;
                        BSDF::eval_pdf(mat, si, wo_local, f_bsdf, bsdf_pdf);
                        L += beta * nee_contribution(scene, si.spawn_ray(es.d), es.dist - 1e-4f,
                                                     es.mesh_id, es, f_bsdf, bsdf_pdf, cos_theta);
                    }
                }
            }

            // ------------------------ BSDF sampling -----------------------

            BSDFSample bs;
            Color3f bsdf_w = BSDF::sample(mat, si, bs, sampler.next2D());

            if (!bs.is_valid()) {
                break;
            }

            // ---- Update loop variables based on current interaction -----

            beta *= bsdf_w;
            eta  *= bs.eta;

            // -------------------- Stopping criterion ---------------------

            const float max_beta = fmaxf(beta.x, fmaxf(beta.y, beta.z));
            if (max_beta <= 0.f) break;

            // Russian roulette stopping probability
            if (depth >= rr_depth) {
                const float rr_prob = fminf(max_beta / (eta * eta), 0.95f);
                if (sampler.next1D() >= rr_prob) break;
                beta = beta / rr_prob;
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


#if !defined(__CUDACC__) && defined(NANOGUI_GLAD)
#include "integrator_ui.h"

FUTABA_NAMESPACE_BEGIN

class PathIntegratorUI : public IntegratorUI {
public:
    std::string getName() const override { return "Path"; }
    int getMode() const override { return INTEGRATOR_PATH; }
};

FUTABA_NAMESPACE_END
#endif