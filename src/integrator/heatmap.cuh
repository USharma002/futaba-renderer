#pragma once

#include "sampler.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"

namespace futaba {

// Heatmap integrator: visualises BVH traversal cost per pixel.
//
// Maps the number of AABB intersection tests for a given ray to a colour
// gradient 
// Useful for diagnosing BVH quality and geometry hot-spots.
//
// The normaliser (kMaxAABBTests = 512) is the AABB-test count that maps to
// full saturation (white). Adjust it to suit your scene's complexity.
struct Heatmap {
    static constexpr int kMaxAABBTests = 512;

    HD Color3f sample(const Ray& ray, const Scene& scene,
                      Sampler& /*sampler*/) const
    {
        const int   aabb_tests = scene.intersectAABBCount(ray, ray.mint, ray.maxt);
        const float normalized = fminf((float)aabb_tests / (float)kMaxAABBTests, 1.f);
        return custom_colormap(normalized);
    }

private:
    HD Color3f custom_colormap(float t) const {
        t = fmaxf(0.f, fminf(1.f, t));

        if (t < 0.25f) {
            const float u = t / 0.25f;
            return Color3f(0.f, 0.f, u * 0.6f);                         // Black -> Dark Blue
        } else if (t < 0.5f) {
            const float u = (t - 0.25f) / 0.25f;
            return Color3f(u * 0.8f, 0.f, 0.6f + u * 0.4f);            // Dark Blue -> Magenta
        } else if (t < 0.75f) {
            const float u = (t - 0.5f) / 0.25f;
            return Color3f(0.8f + u * 0.2f, u * 0.3f, 1.f - u * 0.2f); // Magenta -> Pink
        } else {
            const float u = (t - 0.75f) / 0.25f;
            return Color3f(1.f, 0.3f + u * 0.7f, 0.8f + u * 0.2f);     // Pink -> White
        }
    }
};

} // namespace futaba
