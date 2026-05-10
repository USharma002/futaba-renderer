#pragma once

#include "bsdf_sample.cuh"
#include "sampler.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"
#include <cmath>

namespace futaba {

struct Path {
    int max_depth;
    int rr_depth;

    HD Path(int max_d = 5, int rr_d = 5) : max_depth(max_d), rr_depth(rr_d) {}

    // Accumulate self-emission from the hit surface.
    HD Color3f direct_emission(const Scene&              scene,
                                const SurfaceIntersection& si,
                                const Color3f&             throughput) const
    {
        return throughput * scene.eval_surface_emission(si);
    }

    HD Color3f sample(const Ray& ray, const Scene& scene, Sampler& sampler) const {
        Color3f L(0.f); // Radiance accumulator
        Color3f f(1.f); // Path throughput
        Ray     current_ray = ray;

        for (int depth = 0; depth < max_depth; ++depth) {
            SurfaceIntersection si;
            if (!scene.intersect(current_ray, current_ray.mint, current_ray.maxt, si))
                break;

            // Emission at the hit point (e.g., area lights hit by camera or bounce ray).
            L += direct_emission(scene, si, f);

            // Sample BSDF for the next direction.
            BSDFSample bs;
            // sample_bsdf() returns f*|cos|/pdf (== bs.weight). Use the return value
            // to update throughput; do not also multiply by bs.weight separately.
            const Color3f bsdf_weight = si.sample_bsdf(bs, sampler.next2D());
            if (!bs.is_valid())
                break;

            f *= bsdf_weight;

            // Early exit for zero-throughput paths (avoids wasted bounces).
            const float max_f = fmaxf(f.x, fmaxf(f.y, f.z));
            if (max_f <= 0.f)
                break;

            // Russian roulette: start terminating after rr_depth bounces.
            if (depth >= rr_depth) {
                const float rr_prob = fminf(max_f, 0.95f);
                if (sampler.next1D() >= rr_prob)
                    break;
                f *= 1.f / rr_prob; // Compensate for the surviving paths.
            }

            current_ray = si.spawn_ray(si.to_world(bs.wo));
        }

        return L;
    }
};

} // namespace futaba
