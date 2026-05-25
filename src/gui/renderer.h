#pragma once
#include "hdrfilm.cuh"
#include "perspective.cuh"
#include "scene.cuh"
#include <vector_types.h>


#include "launch_params.h"

namespace futaba {
    class DenoiserManager;
}

// CUDA renderer writes directly into the mapped PBO (zero-copy).
// `scene` must already have triangles/materials uploaded to the GPU.
void launch_render(uchar4 *d_buffer, futaba::HDRFilm *film, int width,
                   int height, const futaba::PerspectiveCamera &camera,
                   const futaba::Scene &scene, int max_depth, int rr_depth,
                   int integrator_mode, int tonemapping_mode, bool use_antialiasing,
                   const ::Vector3f &phong_light_dir,
                   float phong_ambient, float phong_diffuse,
                   float phong_specular, float phong_shininess,
                   bool use_denoiser,
                   futaba::DenoiserManager* denoiser,
                   int path_guiding_mode,
                   // Training buffers
                   float* train_active,
                   Point3f* train_position,
                   Color3f* train_normals,
                   Color3f* train_wi,
                   Color3f* train_wo,
                   Color3f* train_radiance,
                   float* train_material_id,
                   // Visualization parameters
                   uchar4* d_vis_buffer,
                   int vis_depth,
                   int vis_buffer_type,
                   bool vis_active);

// Build the built-in Cornell-box fallback into an existing Scene.
void buildCornellBox(futaba::Scene &scene);