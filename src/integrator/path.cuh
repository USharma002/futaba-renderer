#pragma once

#include "bsdf_sample.cuh"
#include "bsdf.cuh"
#include "emitter_sampler.cuh"
#include "sampler.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"
#include "guiding_device.cuh"
#include "training_buffer.h"
#include "path_recorder.cuh"
#include <cmath>

namespace futaba {

struct Path {
    int             max_depth;
    int             rr_depth;
    EmitterSampler emitter_sampler;

    HD explicit Path(int max_d = 12, int rr_d = 5, EmitterSampler sampler = EmitterSampler())
        : max_depth(max_d), rr_depth(rr_d), emitter_sampler(sampler) {}


    template<bool RecordTraining = false, bool RecordDenoiseGuides = false>
    HD Color3f sample(const Ray& ray, const Scene& scene, Sampler& sampler,
                      const TrainingBuffers& tb = TrainingBuffers(),
                      Color3f* denoise_albedo_buffer = nullptr,
                      Color3f* denoise_normal_buffer = nullptr,
                      int pixel_index = -1,
                      int sample_count = 1,
                      const PerspectiveCamera* camera = nullptr,
                      int guiding_mode = 0,
                      STreeNode* sTreeNodes = nullptr,
                      const AABB& sTreeAABB = AABB(),
                      float bsdf_sampling_fraction = 0.5f,
                      int ppg_distribution_mode = 0) const
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
        constexpr int MAX_LOCAL_DEPTH = RecordTraining ? 16 : 1;
        Color3f local_Le[MAX_LOCAL_DEPTH];
        Color3f local_nee[MAX_LOCAL_DEPTH];
        Color3f local_bsdf[MAX_LOCAL_DEPTH];
        float   local_pdf[MAX_LOCAL_DEPTH];
        float   local_dTreePdf[MAX_LOCAL_DEPTH];

        if constexpr (RecordTraining) {
            if (record_training) {
                for (int i = 0; i < MAX_LOCAL_DEPTH; ++i) {
                    local_Le[i] = Color3f(0.f);
                    local_nee[i] = Color3f(0.f);
                    local_bsdf[i] = Color3f(0.f);
                    local_pdf[i] = 0.f;
                    local_dTreePdf[i] = 0.f;
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
            PathRecorder::record_denoiser_guides<RecordDenoiseGuides>(si, hit, depth, current_ray,
                                                                     denoise_albedo_buffer, denoise_normal_buffer,
                                                                     pixel_index, sample_count, camera, scene);

            // Record training features for this bounce
            PathRecorder::record_training_features<RecordTraining>(si, hit, depth, current_ray, tb, record_training, guiding_mode, scene.materials);

            // ---------------------- Direct emission ----------------------

            Color3f Le = emitter_sampler.eval_direct_emission(scene, si, hit, current_ray, prev_p, prev_bsdf_pdf, prev_bsdf_delta);
            L += beta * Le;

            if constexpr (RecordTraining) {
                if (record_training && depth < MAX_LOCAL_DEPTH) {
                    local_Le[depth] = Le;
                }
            }

            if (!hit)
                break;

            const Material& mat = scene.materials[si.material_id];

            // ---------------------- Emitter sampling ----------------------

            Color3f nee_val = emitter_sampler.sample_direct_emitter(scene, si, mat, sampler, guiding_mode, sTreeNodes, sTreeAABB, bsdf_sampling_fraction);
            L += beta * nee_val;

            if constexpr (RecordTraining) {
                if (record_training && depth < MAX_LOCAL_DEPTH) {
                    local_nee[depth] = nee_val;
                }
            }

            // ------------------ BSDF sampling (guided) -------------------

            BSDFSample bs;
            Color3f bsdf_w(0.f);
            float dTreePdf = 0.f;

            GuidingDistribution guiding(guiding_mode, si, sTreeNodes, sTreeAABB);
            bsdf_w = guiding.sample_guided(mat, si, bs, sampler, bsdf_sampling_fraction, dTreePdf);

            if (!bs.is_valid()) {
                if constexpr (RecordTraining) {
                    if (record_training && depth < tb.max_depth) {
                        int buf_idx = depth * tb.img_size + tb.pixel_index;
                        tb.active[buf_idx] = 0.0f;
                    }
                }
                break;
            }

            // ---- Update loop variables based on current interaction -----

            beta *= bsdf_w;
            eta  *= bs.eta;

            // Record training features for BSDF sample
            if constexpr (RecordTraining) {
                if (record_training && depth < MAX_LOCAL_DEPTH) {
                    local_bsdf[depth] = bsdf_w * bs.pdf;
                    local_pdf[depth] = bs.pdf;
                    local_dTreePdf[depth] = dTreePdf;
                }
                if (record_training && depth < tb.max_depth && tb.wo) {
                    if (guiding_mode == PATH_GUIDING_PPG || guiding_mode == PATH_GUIDING_NPM) {
                        int buf_idx = depth * tb.img_size + tb.pixel_index;
                        tb.wo[buf_idx] = (guiding_mode == PATH_GUIDING_PPG) ? si.to_world(bs.wo) : bs.wo;
                    }
                }
            }

            // -------------------- Stopping criterion ---------------------

            const float max_beta = fmaxf(beta.x, fmaxf(beta.y, beta.z));
            if (max_beta <= 0.f) break;

            // Russian roulette stopping probability
            if (depth >= rr_depth) {
                const float rr_prob = fminf(max_beta * eta * eta, 0.95f);
                if (sampler.next1D() >= rr_prob) break;
                beta = beta / rr_prob;
            }

            // Advance training buffer state
            if constexpr (RecordTraining) {
                if (record_training && depth < tb.max_depth && tb.wi && guiding_mode == PATH_GUIDING_PPG) {
                    int buf_idx = depth * tb.img_size + tb.pixel_index;
                    tb.wi[buf_idx] = beta;
                }
            }

            prev_p          = si.p;
            prev_bsdf_pdf   = bs.pdf;
            prev_bsdf_delta = BSDF::is_delta(mat.type);
            current_ray     = si.spawn_ray(si.to_world(bs.wo));
        }

        // Backward propagation of radiance from local register arrays
        // Backward propagation of radiance from local register arrays
        PathRecorder::backward_propagate_radiance<RecordTraining>(final_depth, tb, record_training, guiding_mode,
                                                                 local_Le, local_nee, local_bsdf, local_pdf,
                                                                 local_dTreePdf, ppg_distribution_mode);
                                                    
        return L;
    }
};

using PathIntegrator = Path;

} // namespace futaba


#if !defined(__CUDACC__) && defined(NANOGUI_GLAD)
#include "integrator_ui.h"

namespace futaba {

class PathIntegratorUI : public IntegratorUI {
public:
    std::string getName() const override { return "Path"; }
    int getMode() const override { return INTEGRATOR_PATH; }
};

} // namespace futaba
#endif
