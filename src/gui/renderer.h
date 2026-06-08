#pragma once
#include "hdrfilm.cuh"
#include "perspective.cuh"
#include "scene.cuh"
#include <vector_types.h>
#include <atomic>

#include "launch_params.h"

namespace futaba {
    class DenoiserManager;

    // Global atomic variables for tracking background OptiX compilation progress
    extern std::atomic<float> g_optixCompileProgress;
    extern std::atomic<const char*> g_optixCompileStatus;
    extern std::atomic<bool> g_optixCompileCompleted;

    // Trigger OptiX pipeline initialization synchronously on the calling thread
    void launch_initial_pipeline_compile();

    // Explicitly clean up the global OptiX pipeline and related GPU buffers
    void cleanup_pipeline();
}

// CUDA renderer writes directly into the mapped PBO (zero-copy).
// `scene` must already have triangles/materials uploaded to the GPU.
void launch_render(futaba::HDRFilm *film,
                   futaba::DenoiserManager* denoiser,
                   futaba::LaunchParams params);

// Build the built-in Cornell-box fallback into an existing Scene.
void buildCornellBox(futaba::Scene &scene);