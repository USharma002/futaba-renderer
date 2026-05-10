#pragma once

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include "types.cuh"

namespace futaba {

#ifndef FUTABA_CUDA_CHECK
#define FUTABA_CUDA_CHECK(call)                                                \
    do {                                                                       \
        const cudaError_t _err = (call);                                       \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_err));             \
        }                                                                      \
    } while (0)
#endif

struct EnvironmentMapEmitter {
    Color3f*  pixels      = nullptr;
    uint32_t  width       = 0;
    uint32_t  height      = 0;
    Matrix4f  toWorld;
    bool      hasMap      = false;
    bool      hasConstant = false;
    Color3f   constant    = Color3f(0.f);

    HD bool isActive() const {
        return hasConstant || (hasMap && pixels != nullptr && width > 0 && height > 0);
    }

    HD Color3f eval(const Vector3f& dirWorld) const {
        if (hasConstant)
            return constant;

        if (!hasMap || pixels == nullptr || width == 0 || height == 0)
            return Color3f(0.f);

        const Vector3f d = normalize(Vector3f(
            toWorld.m[0][0] * dirWorld.x + toWorld.m[1][0] * dirWorld.y + toWorld.m[2][0] * dirWorld.z,
            toWorld.m[0][1] * dirWorld.x + toWorld.m[1][1] * dirWorld.y + toWorld.m[2][1] * dirWorld.z,
            toWorld.m[0][2] * dirWorld.x + toWorld.m[1][2] * dirWorld.y + toWorld.m[2][2] * dirWorld.z
        ));

        const float phi = atan2f(d.y, d.x);
        float u = phi / (2.f * M_PI);
        if (u < 0.f) u += 1.f;
        const float v = acosf(fmaxf(-1.f, fminf(1.f, d.z))) / M_PI;

        const float x = u * (float)width  - 0.5f;
        const float y = v * (float)height - 0.5f;

        const int x0 = (int)floorf(x);
        const int y0 = (int)floorf(y);
        const float tx = x - (float)x0;
        const float ty = y - (float)y0;

        auto wrap_x = [this](int xi) -> int {
            const int w = (int)width;
            xi %= w;
            if (xi < 0) xi += w;
            return xi;
        };

        const int x00 = wrap_x(x0);
        const int x10 = wrap_x(x0 + 1);
        const int y00 = (int)fminf(fmaxf(y0, 0), (int)height - 1);
        const int y10 = (int)fminf(fmaxf(y0 + 1, 0), (int)height - 1);

        const Color3f c00 = pixels[y00 * width + x00];
        const Color3f c10 = pixels[y00 * width + x10];
        const Color3f c01 = pixels[y10 * width + x00];
        const Color3f c11 = pixels[y10 * width + x10];

        const Color3f c0 = c00 * (1.f - tx) + c10 * tx;
        const Color3f c1 = c01 * (1.f - tx) + c11 * tx;
        return c0 * (1.f - ty) + c1 * ty;
    }

    HD Vector3f sampleDirection(const Point2f& sample) const {
        const float z = 1.f - 2.f * sample.x;
        const float r = sqrtf(fmaxf(0.f, 1.f - z * z));
        const float phi = 2.f * M_PI * sample.y;
        return Vector3f(r * cosf(phi), r * sinf(phi), z);
    }

    HD float pdf(const Vector3f& /*dirWorld*/) const {
        return isActive() ? 1.f / (4.f * M_PI) : 0.f;
    }

    HD Color3f sample(const Point2f& sample, Vector3f& dirWorld, float& outPdf) const {
        dirWorld = sampleDirection(sample);
        outPdf = pdf(dirWorld);
        return eval(dirWorld);
    }

    void setMap(const Color3f* hostPixels, uint32_t mapWidth, uint32_t mapHeight, const Matrix4f& envToWorld) {
        if (pixels != nullptr) {
            FUTABA_CUDA_CHECK(cudaFree(pixels));
            pixels = nullptr;
        }

        width = mapWidth;
        height = mapHeight;
        toWorld = envToWorld;
        hasMap = (width > 0 && height > 0 && hostPixels != nullptr);
        hasConstant = false;
        constant = Color3f(0.f);

        if (!hasMap)
            return;

        FUTABA_CUDA_CHECK(cudaMalloc(&pixels, width * height * sizeof(Color3f)));
        FUTABA_CUDA_CHECK(cudaMemcpy(pixels, hostPixels,
                                     width * height * sizeof(Color3f),
                                     cudaMemcpyHostToDevice));
    }

    void setConstant(const Color3f& radiance) {
        if (pixels != nullptr) {
            FUTABA_CUDA_CHECK(cudaFree(pixels));
            pixels = nullptr;
        }
        width = 0;
        height = 0;
        hasMap = false;
        hasConstant = true;
        constant = radiance;
    }

    void clear() {
        if (pixels != nullptr) {
            FUTABA_CUDA_CHECK(cudaFree(pixels));
            pixels = nullptr;
        }
        width = 0;
        height = 0;
        hasMap = false;
        hasConstant = false;
        constant = Color3f(0.f);
    }
};

} // namespace futaba
