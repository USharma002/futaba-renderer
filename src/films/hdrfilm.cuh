#pragma once

#include "types.cuh"
#include <cuda_runtime.h>

namespace futaba {
class Bitmap;

class HDRFilm {
public:
    int width, height;
    Color3f* d_pixels; // Pointer to GPU (Device) memory!
    int sampleCount;   // How many frames we have accumulated

    HDRFilm(int width, int height) : width(width), height(height), sampleCount(0) {
        // Allocate memory on the GPU
        cudaMalloc(&d_pixels, width * height * sizeof(Color3f));
        clear();
    }

    ~HDRFilm() {
        cudaFree(d_pixels);
    }

    // Wipes the film clean (e.g., when the camera moves)
    void clear() {
        // Instantly sets all GPU memory to 0
        cudaMemset(d_pixels, 0, width * height * sizeof(Color3f));
        sampleCount = 0;
    }

    // Copies GPU pixels to a CPU Bitmap, dividing by sampleCount
    Bitmap* toBitmap() const;
};

} // namespace futaba