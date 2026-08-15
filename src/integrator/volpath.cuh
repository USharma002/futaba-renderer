#pragma once

#include "bsdf_sample.cuh"
#include "bsdf.cuh"
#include "emitter_sampler.cuh"
#include "medium/medium.cuh"
#include "phase/phase.cuh"
#include "sampler.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"
#include <cmath>

FUTABA_NAMESPACE_BEGIN

struct VolPath {
    int            max_depth;
    int            rr_depth;
    EmitterSampler emitter_sampler;

    HD explicit VolPath(int max_d = 32, int rr_d = 5, EmitterSampler sampler = EmitterSampler())
        : max_depth(max_d), rr_depth(rr_d), emitter_sampler(sampler) {}

    // Evaluate beam transmittance and visibility along shadow ray through media and null boundaries
    HD static Color3f eval_shadow_transmittance(const Scene& scene, const Point3f& origin, const Vector3f& d,
                                                float max_dist, int start_medium_id, int target_mesh_id)
    {
        Color3f tr(1.0f);
        int med_id = start_medium_id;
        float t_min = 1e-4f;
        float t_max = max_dist;
        Point3f ray_o = origin;

        for (int step = 0; step < 16; ++step) {
            Ray test_ray(ray_o, d, t_min, t_max);
            SurfaceInteraction si;
            bool hit = scene.intersect(test_ray, t_min, t_max, si);

            const Medium* med = (med_id >= 0 && (uint32_t)med_id < scene.mediumCount) ? &scene.media[med_id] : nullptr;
            float seg_dist = hit ? si.t : t_max;

            if (med && med->is_active()) {
                tr *= med->eval_transmittance(seg_dist);
                if (tr.maxComponent() < 1e-6f) return Color3f(0.f);
            }

            if (!hit) break;

            if (si.shape_id == target_mesh_id) break;

            const Material& mat = scene.materials[si.material_id];
            if (mat.type == BSDF_ID_NULL) {
                bool entering = (dot(si.n, d) < 0.f);
                med_id = entering ? mat.interiorMediumId : mat.exteriorMediumId;
                ray_o = si.p;
                t_max -= seg_dist;
                if (t_max <= 1e-4f) break;
            } else {
                return Color3f(0.f);
            }
        }

        return tr;
    }

    HD Color3f sample(const Ray& ray, const Scene& scene, Sampler& sampler) const {
        Ray     current_ray = ray;
        Color3f L(0.f), beta(1.f);
        float   eta = 1.f;
        int     current_medium_id = -1; // Starts in vacuum

        Point3f prev_p          = ray.o;
        float   prev_pdf        = 1.f;
        bool    prev_delta      = true;

        for (int depth = 0; depth < max_depth; ++depth) {
            SurfaceInteraction si;
            bool hit = scene.intersect(current_ray, current_ray.mint, current_ray.maxt, si);
            float max_t = hit ? si.t : Infinity;

            const Medium* current_medium = (current_medium_id >= 0 && (uint32_t)current_medium_id < scene.mediumCount)
                ? &scene.media[current_medium_id] : nullptr;

            MediumInteraction mi;
            Color3f tr_weight(1.f);
            float med_pdf = 1.f;
            bool is_medium_scatter = false;

            if (current_medium && current_medium->is_active()) {
                int channel = (int)(sampler.next1D() * 3.f);
                if (channel > 2) channel = 2;
                is_medium_scatter = current_medium->sample_distance(current_ray, max_t, sampler.next1D(), channel, mi, tr_weight, med_pdf);
            }

            beta *= tr_weight;

            if (is_medium_scatter) {
                // Medium In-Scattering: Direct lighting (NEE)
                if (scene.use_nee) {
                    DirectionSample ds;
                    Color3f em_weight;
                    Point3f u3(sampler.next1D(), sampler.next1D(), sampler.next1D());
                    SurfaceInteraction ref_si;
                    ref_si.p = mi.p;

                    if (emitter_sampler.sample_direction(scene, ref_si, u3, ds, em_weight)) {
                        float shadow_dist = ds.dist - 1e-4f;
                        Color3f shadow_tr = eval_shadow_transmittance(scene, mi.p, ds.d, shadow_dist, current_medium_id, ds.mesh_id);

                        if (shadow_tr.maxComponent() > 0.f) {
                            float p_val = Phase::eval(current_medium->g, mi.wo, ds.d);
                            float mis = ds.delta ? 1.f : mis_weight(ds.pdf, p_val);
                            L += beta * p_val * ds.Le * shadow_tr * (mis / ds.pdf);
                        }
                    }
                }

                // Sample Phase function
                PhaseSample ps;
                Vector3f wo = Phase::sample(current_medium->g, mi.wo, sampler.next2D(), ps);
                if (ps.pdf <= 0.f) break;

                prev_p      = mi.p;
                prev_pdf    = ps.pdf;
                prev_delta  = false;
                current_ray = Ray(mi.p, wo, 1e-4f, Infinity);
            } else {
                // Surface Interaction or Environment Miss
                Color3f Le = emitter_sampler.eval_direct_emission(scene, si, hit, current_ray, prev_p, prev_pdf, prev_delta);
                L += beta * Le;

                if (!hit) break;

                const Material& mat = scene.materials[si.material_id];

                // Medium boundary transition (null BSDF interface)
                if (mat.type == BSDF_ID_NULL) {
                    bool entering = (dot(si.n, current_ray.d) < 0.f);
                    current_medium_id = entering ? mat.interiorMediumId : mat.exteriorMediumId;
                    current_ray = si.spawn_ray(current_ray.d);
                    prev_delta = true;
                    continue;
                }

                const bool scatterable = !BSDF::is_delta(mat.type);

                // Surface Emitter Sampling (NEE)
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

                            float shadow_dist = ds.dist - 1e-4f;
                            Point3f shadow_o = si.spawn_ray(ds.d).o;
                            Color3f shadow_tr = eval_shadow_transmittance(scene, shadow_o, ds.d, shadow_dist, current_medium_id, ds.mesh_id);

                            if (shadow_tr.maxComponent() > 0.f) {
                                float mis = ds.delta ? 1.f : mis_weight(ds.pdf, bsdf_pdf);
                                L += beta * f_bsdf * ds.Le * shadow_tr * (cos_theta * mis / ds.pdf);
                            }
                        }
                    }
                }

                // Sample BSDF
                BSDFSample bs;
                Color3f bsdf_w = BSDF::sample(mat, si, bs, sampler.next2D());
                if (!bs.is_valid()) break;

                // Update medium if transmitting through boundary
                if (bs.is_transmissive()) {
                    bool entering = (dot(si.n, current_ray.d) < 0.f);
                    current_medium_id = entering ? mat.interiorMediumId : mat.exteriorMediumId;
                }

                beta *= bsdf_w;
                eta  *= bs.eta;

                prev_p          = si.p;
                prev_pdf        = bs.pdf;
                prev_delta      = BSDF::is_delta(mat.type);
                current_ray     = si.spawn_ray(si.to_world(bs.wo));
            }

            // Russian roulette
            const float max_beta = beta.maxComponent();
            if (max_beta <= 0.f) break;

            if (depth >= rr_depth) {
                const float rr_prob = fminf(fmaxf(max_beta / (eta * eta), 0.05f), 0.95f);
                if (sampler.next1D() >= rr_prob) break;
                beta /= rr_prob;
            }
        }

        return L;
    }
};

FUTABA_NAMESPACE_END
