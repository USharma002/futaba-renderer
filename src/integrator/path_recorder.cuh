#pragma once

#include "types.cuh"
#include "surface_interaction.cuh"
#include "ray.cuh"
#include "perspective.cuh"
#include "scene.cuh"
#include "material.cuh"
#include "bsdf.cuh"

FUTABA_NAMESPACE_BEGIN

// Sidecar object the path integrator reports intersections to, at every
// bounce. It owns the buffers it writes into and decides on its own what
// (if anything) is worth recording — the integrator doesn't need to know
// what's being collected or why.
//
// Currently only fills the denoiser's albedo/normal guide buffers at the
// first bounce. If we later want to grab more data (a training buffer,
// bounce statistics, etc.), it belongs here as another field + branch in
// record(), not as new params threaded through Path::sample.
struct PathRecorder {
    Color3f* albedo_buffer = nullptr;
    Color3f* normal_buffer = nullptr;
    int      pixel_index   = -1;
    int      sample_count  = 1;
    const PerspectiveCamera* camera = nullptr;

    HD void record(int depth, bool hit, const SurfaceIntersection& si,
                   const Ray& current_ray, const Scene& scene) const
    {
        if (depth != 0 || !albedo_buffer || !normal_buffer)
            return;

        Color3f  alb(0.f);
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
            albedo_buffer[pixel_index] = alb;
            normal_buffer[pixel_index] = norm;
        } else {
            albedo_buffer[pixel_index] += alb;
            normal_buffer[pixel_index] += norm;
        }
    }
};

FUTABA_NAMESPACE_END