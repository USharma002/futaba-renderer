#pragma once

#include "path.cuh"
#include "sampler.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"
#include "emitter_sampler.cuh"
#include <cmath>

namespace futaba {

struct VolumetricPath {
    int  max_depth;
    int  rr_depth;
    bool hide_emitters;
    EmitterSampler emitter_sampler;

    HD VolumetricPath(int max_d = 8, int rr_d = 5, bool hide_e = false, EmitterSampler sampler = EmitterSampler())
        : max_depth(max_d), rr_depth(rr_d), hide_emitters(hide_e), emitter_sampler(sampler) {}

    HD static float mis_weight(float pdf_a, float pdf_b) {
        const float a2 = pdf_a * pdf_a;
        const float b2 = pdf_b * pdf_b;
        const float d  = a2 + b2;
        if (d <= 0.f)
            return 0.f;
        const float w = a2 / d;
        return isfinite(w) ? w : 0.f;
    }

    HD Color3f sample(const Ray& ray, const Scene& scene, Sampler& sampler) const {
        Color3f L(0.f), beta(1.f);
        float   eta = 1.f;
        Ray     current_ray = ray;

        float   prev_bsdf_pdf   = 1.f;
        bool    prev_bsdf_delta = true;
        Point3f prev_p          = ray.o;

        bool inside_medium = false;

        for (int depth = 0; depth < max_depth; ++depth) {
            SurfaceIntersection si;
            bool hit = scene.intersect(current_ray, current_ray.mint, current_ray.maxt, si);

            if (scene.hasMedium && inside_medium) {
                float sigma_t_c = 0.f;
                scene.medium.sampleChannel(sampler.next1D(), sigma_t_c);

                float t = -logf(1.f - fminf(sampler.next1D(), 0.999999f)) / fmaxf(sigma_t_c, 1e-6f);

                if (!hit || t < si.t) {
                    Point3f p_t = current_ray.o + t * current_ray.d;

                    Color3f transmittance = scene.medium.evalTransmittance(t);

                    float pdf = sigma_t_c * expf(-sigma_t_c * t);
                    if (pdf <= 0.f) return L;

                    Color3f W = (transmittance * scene.medium.homogeneous.sigmaS) / pdf;
                    beta *= W;

                    if (scene.use_nee) {
                        EmitterSample es;
                        Point3f u3(sampler.next1D(), sampler.next1D(), sampler.next1D());
                        SurfaceIntersection vol_si;
                        vol_si.p = p_t;
                        vol_si.wi = -current_ray.d;
                        vol_si.front_face = true;
                        
                        if (emitter_sampler.sample(scene, vol_si, u3, es) && es.pdf > 0.f) {
                            if (es.Le.x > 0.f || es.Le.y > 0.f || es.Le.z > 0.f) {
                                Ray shadow(p_t, es.d);
                                float t_max = es.dist - 1e-4f;
                                bool visible = !scene.occluded(shadow, 1e-6f, t_max, es.mesh_id);
                                if (visible) {
                                    Color3f trans_light = scene.medium.evalTransmittance(es.dist);

                                    float phase_val = scene.medium.getPhaseFunction().eval(dot(current_ray.d, es.d));
                                    float weight_nee = es.delta ? 1.f : mis_weight(es.pdf, phase_val);
                                    L += beta * (phase_val / es.pdf) * es.Le * trans_light * weight_nee;
                                }
                            }
                        }
                    }

                    Vector3f wo;
                    float phase_pdf = 0.f;
                    scene.medium.getPhaseFunction().sample(current_ray.d, sampler.next2D(), wo, phase_pdf);

                    current_ray = Ray(p_t, wo);
                    prev_p = p_t;
                    prev_bsdf_pdf = phase_pdf;
                    prev_bsdf_delta = false;

                    continue;
                } else {
                    Color3f transmittance = scene.medium.evalTransmittance(si.t);
                    float prob_surf = expf(-sigma_t_c * si.t);
                    if (prob_surf > 0.f) {
                        beta *= transmittance / prob_surf;
                    }
                }
            }

            if (hit && si.mat_type == BSDF_ID_NULL) {
                inside_medium = !inside_medium;
                current_ray = si.spawn_ray(current_ray.d);
                prev_bsdf_delta = true;
                depth--;
                continue;
            }

            if (!hit) {
                const Color3f Le = scene.eval_environment(current_ray.d);
                if (Le.x > 0.f || Le.y > 0.f || Le.z > 0.f) {
                    float w = 1.f;
                    if (scene.use_nee && !prev_bsdf_delta) {
                        w = mis_weight(prev_bsdf_pdf,
                                       emitter_sampler.pdf(scene, -1, current_ray.d, 1e30f));
                    }
                    L += beta * w * Le;
                }
                break;
            }

            Color3f emission = scene.eval_surface_emission(si);
            if (emission.x > 0.f || emission.y > 0.f || emission.z > 0.f) {
                float w = 1.f;
                if (scene.use_nee && !prev_bsdf_delta) {
                    const Vector3f d = normalize(si.p - prev_p);
                    const float dist = length(si.p - prev_p);
                    w = mis_weight(prev_bsdf_pdf,
                                   emitter_sampler.pdf(scene, si.primitive_id, d, dist));
                }
                L += beta * w * emission;
            }

            if (scene.use_nee && !si.is_bsdf_delta()) {
                EmitterSample es;
                Point3f u3(sampler.next1D(), sampler.next1D(), sampler.next1D());
                if (emitter_sampler.sample(scene, si, u3, es) && es.pdf > 0.f) {
                    if (es.Le.x > 0.f || es.Le.y > 0.f || es.Le.z > 0.f) {
                        const Vector3f wo_local = si.to_local(es.d);
                        const float cos_s = wo_local.z;
                        if (cos_s > 0.f) {
                            Color3f f_bsdf; float pdf_bsdf;
                            si.eval_pdf_bsdf(wo_local, f_bsdf, pdf_bsdf);
                            if (f_bsdf.x > 0.f || f_bsdf.y > 0.f || f_bsdf.z > 0.f) {
                                Ray shadow = si.spawn_ray(es.d);
                                float t_max = es.dist - 1e-4f;
                                bool visible = !scene.occluded(shadow, 1e-6f, t_max, es.mesh_id);
                                if (visible) {
                                    Color3f trans_light(1.f);
                                    if (scene.hasMedium && inside_medium) {
                                        trans_light = scene.medium.evalTransmittance(es.dist);
                                    }
                                    const float w = es.delta ? 1.f : mis_weight(es.pdf, pdf_bsdf);
                                    L += beta * f_bsdf * (cos_s / es.pdf) * es.Le * trans_light * w;
                                }
                            }
                        }
                    }
                }
            }

            BSDFSample bs;
            Color3f f_bsdf = si.sample_bsdf(bs, sampler.next2D());
            if (f_bsdf.x <= 0.f && f_bsdf.y <= 0.f && f_bsdf.z <= 0.f) break;
            if (!bs.is_valid()) break;

            current_ray = si.spawn_ray(si.to_world(bs.wo));
            eta *= bs.eta;
            beta *= f_bsdf;
            prev_p = si.p;
            prev_bsdf_pdf = bs.pdf;
            prev_bsdf_delta = si.is_bsdf_delta();

            float beta_max = fmaxf(beta.x, fmaxf(beta.y, beta.z));
            if (beta_max <= 0.f) break;

            float rr_prob = fminf(beta_max * eta * eta, 0.95f);
            if (depth >= rr_depth) {
                beta *= 1.f / rr_prob;
                if (sampler.next1D() >= rr_prob) break;
            }
        }

        return L;
    }
};

} // namespace futaba

#if !defined(__CUDACC__) && defined(NANOGUI_GLAD)
#include "integrator_ui.h"

namespace futaba {

class VolPathIntegratorUI : public IntegratorUI {
public:
    std::string getName() const override { return "VolPath"; }
    int getMode() const override { return INTEGRATOR_VOLPATH; }
};

} // namespace futaba
#endif

