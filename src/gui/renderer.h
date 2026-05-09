#pragma once
#include "hdrfilm.cuh"
#include "perspective.cuh"
#include "scene.cuh"
#include <vector_types.h>


#include "launch_params.h"

// CUDA renderer writes directly into the mapped PBO (zero-copy).
// `scene` must already have triangles/materials uploaded to the GPU.
void launch_render(uchar4 *d_buffer, futaba::HDRFilm *film, int width,
                   int height, const futaba::PerspectiveCamera &camera,
                   const futaba::Scene &scene, int max_depth, int rr_depth,
                   int integrator_mode, bool use_antialiasing);

// Build the built-in Cornell-box fallback into an existing Scene.
void buildCornellBox(futaba::Scene &scene);