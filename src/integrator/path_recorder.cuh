#pragma once

#include "types.cuh"
#include "surface_interaction.cuh"
#include "ray.cuh"
#include "training_buffer.h"
#include "perspective.cuh"
#include "scene.cuh"
#include "material.cuh"
#include "bsdf.cuh"

namespace futaba {

struct PathRecorder {
    template<bool RecordDenoiseGuides>
    HD static void record_denoiser_guides(const SurfaceIntersection& si, bool hit, int depth, const Ray& current_ray,
                                          Color3f* denoise_albedo_buffer, Color3f* denoise_normal_buffer,
                                          int pixel_index, int sample_count, const PerspectiveCamera* camera,
                                          const Scene& scene)
    {
        if constexpr (RecordDenoiseGuides) {
            if (depth == 0) {
                Color3f alb(0.f);
                Vector3f norm(0.f);
                if (hit) {
                    const Material& mat = scene.materials[si.material_id];
                    alb = BSDF::get_albedo(mat, si);
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

    template<bool RecordTraining>
    HD static void record_training_features(const SurfaceIntersection& si, bool hit, int depth, const Ray& current_ray,
                                            const TrainingBuffers& tb, bool record_training, int guiding_mode,
                                            const Material* materials)
    {
        if constexpr (RecordTraining) {
            if (record_training && depth < tb.max_depth) {
                int buf_idx = depth * tb.img_size + tb.pixel_index;
                tb.active[buf_idx] = 1.0f;

                if (hit) {
                    const Material& mat = materials[si.material_id];
                    if (guiding_mode == PATH_GUIDING_PPG && (BSDF::is_delta(mat.type) || mat.type == BSDF_ID_NULL)) {
                        tb.active[buf_idx] = 0.0f;
                    } else if (guiding_mode == PATH_GUIDING_PPG || guiding_mode == PATH_GUIDING_NPM) {
                        if (tb.position) tb.position[buf_idx] = si.p;
                    }
                    if (guiding_mode == PATH_GUIDING_NPM) {
                        if (tb.normals) tb.normals[buf_idx] = si.n;
                        if (tb.wi) tb.wi[buf_idx] = si.to_local(-current_ray.d);
                        if (tb.material_id) tb.material_id[buf_idx] = (float)si.material_id;
                    }
                } else {
                    tb.active[buf_idx] = 0.0f;
                    if (guiding_mode == PATH_GUIDING_NPM) {
                        if (tb.material_id) tb.material_id[buf_idx] = -1.0f;
                    }
                }
            }
        }
    }

    template<bool RecordTraining>
    HD static void backward_propagate_radiance(int final_depth, const TrainingBuffers& tb, bool record_training, int guiding_mode,
                                               const Color3f* local_Le, const Color3f* local_nee,
                                               const Color3f* local_bsdf, const float* local_pdf,
                                               const float* local_dTreePdf, int ppg_distribution_mode)
    {
        if constexpr (RecordTraining) {
            if (record_training && tb.radiance && (guiding_mode == PATH_GUIDING_PPG || guiding_mode == PATH_GUIDING_NPM)) {
                Color3f incident = Color3f(0.f);
                constexpr int MAX_LOCAL_DEPTH = 16;
                int start_depth = (final_depth < MAX_LOCAL_DEPTH) ? final_depth - 1 : MAX_LOCAL_DEPTH - 1;
                for (int d = start_depth; d >= 0; --d) {
                    Color3f outgoing = local_Le[d] + local_nee[d] + local_bsdf[d] * incident;
                    if (d < tb.max_depth) {
                        int buf_idx = d * tb.img_size + tb.pixel_index;
                        if (tb.active[buf_idx] > 0.5f) {
                            if (guiding_mode == PATH_GUIDING_PPG) {
                                if (ppg_distribution_mode == 3) {
                                    tb.radiance[buf_idx] = outgoing;
                                } else {
                                    float val = 0.f;
                                    if (ppg_distribution_mode == 0) {
                                        val = (outgoing.x + outgoing.y + outgoing.z) / 3.f;
                                    } else {
                                        Color3f prod = outgoing * local_bsdf[d];
                                        float prod_val = (prod.x + prod.y + prod.z) / 3.f;
                                        if (local_pdf[d] > 0.f) {
                                            float mis_weight = local_dTreePdf[d] / local_pdf[d];
                                            prod_val *= mis_weight;
                                        }
                                        if (ppg_distribution_mode == 2) {
                                            val = prod_val * prod_val;
                                        } else {
                                            val = prod_val;
                                        }
                                    }
                                    tb.radiance[buf_idx] = Color3f(val);
                                }
                            } else {
                                tb.radiance[buf_idx] = outgoing;
                            }
                            if (tb.direction_pdf) {
                                tb.direction_pdf[buf_idx] = local_pdf[d];
                            }
                        } else {
                            tb.radiance[buf_idx] = Color3f(0.f);
                            if (tb.direction_pdf) {
                                tb.direction_pdf[buf_idx] = 0.f;
                            }
                        }
                    }
                    incident = outgoing;
                }
            }
        }
    }
};

} // namespace futaba
