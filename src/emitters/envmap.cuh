#pragma once

#include <cstdint>
#include <vector>
#include "types.cuh"
#include "common.cuh"
#include "warp.cuh"
#include "distribution.cuh"

FUTABA_NAMESPACE_BEGIN


struct EnvironmentMapEmitter {
    Color3f*  pixels      = nullptr;
    uint32_t  width       = 0;
    uint32_t  height      = 0;
    Matrix4f  toWorld;
    bool      hasMap      = false;
    bool      hasConstant = false;
    Color3f   constant    = Color3f(0.f);

    // 2D Importance Sampling Tables (GPU Device pointers)
    float*    condCdfs    = nullptr; // Size: height * (width + 1)
    float*    marginalCdf = nullptr; // Size: height + 1
    float     marginalSum = 0.f;

    HD bool isActive() const {
        return hasConstant || (hasMap && pixels != nullptr && width > 0 && height > 0);
    }

    // Evaluate the environment map in the given world direction.
    HD Color3f eval(const Vector3f& dirWorld) const {
        if (hasConstant)
            return constant;

        if (!hasMap || pixels == nullptr || width == 0 || height == 0)
            return Color3f(0.f);

        // Convert world direction to environment map UV coordinates
        const Vector3f d = normalize(Vector3f(
            toWorld.m[0][0] * dirWorld.x + toWorld.m[1][0] * dirWorld.y + toWorld.m[2][0] * dirWorld.z,
            toWorld.m[0][1] * dirWorld.x + toWorld.m[1][1] * dirWorld.y + toWorld.m[2][1] * dirWorld.z,
            toWorld.m[0][2] * dirWorld.x + toWorld.m[1][2] * dirWorld.y + toWorld.m[2][2] * dirWorld.z
        ));

        const float phi = atan2f(d.x, -d.z);
        float u = phi * INV_TWOPI;
        if (u < 0.f) u += 1.f;
        const float v = acosf(clamp(d.y, -1.f, 1.f)) * INV_PI;

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
        const int max_y = (int)height - 1;
        const int y00 = (y0 < 0) ? 0 : ((y0 > max_y) ? max_y : y0);
        const int y10 = ((y0 + 1) < 0) ? 0 : (((y0 + 1) > max_y) ? max_y : (y0 + 1));

        const Color3f c00 = pixels[y00 * width + x00];
        const Color3f c10 = pixels[y00 * width + x10];
        const Color3f c01 = pixels[y10 * width + x00];
        const Color3f c11 = pixels[y10 * width + x10];

        const Color3f c0 = c00 * (1.f - tx) + c10 * tx;
        const Color3f c1 = c01 * (1.f - tx) + c11 * tx;
        return c0 * (1.f - ty) + c1 * ty;
    }

    // Sample local direction on the unit sphere proportional to environment map luminance.
    HD Vector3f sampleDirection(const Point2f& sample) const {
        if (hasConstant) {
            return Warp::squareToUniformSphere(sample);
        }

        if (!hasMap || pixels == nullptr || width == 0 || height == 0 || marginalSum <= 0.f) {
            return Vector3f(0.f);
        }

        // 1. Sample row (v coordinate) using marginal CDF
        int v = findIntervalDevice(marginalCdf, height, sample.y);
        float cdf_v0 = marginalCdf[v];
        float cdf_v1 = marginalCdf[v + 1];
        float dv = (sample.y - cdf_v0) / (cdf_v1 - cdf_v0);
        if (cdf_v1 - cdf_v0 <= 0.f) dv = 0.f;
        float continuous_v = (v + dv) / height;

        // 2. Sample column (u coordinate) using row conditional CDF
        const float* rowCdf = condCdfs + (size_t)v * (width + 1);
        int u = findIntervalDevice(rowCdf, width, sample.x);
        float cdf_u0 = rowCdf[u];
        float cdf_u1 = rowCdf[u + 1];
        float du = (sample.x - cdf_u0) / (cdf_u1 - cdf_u0);
        if (cdf_u1 - cdf_u0 <= 0.f) du = 0.f;
        float continuous_u = (u + du) / width;

        // 3. Map normalized coordinates (u, v) to spherical angles (theta, phi)
        float theta = continuous_v * M_PI;
        float phi = continuous_u * 2.f * M_PI;

        float sinTheta = sinf(theta);
        float cosTheta = cosf(theta);

        // Convert to local direction (consistent with eval uv coordinate mapping)
        Vector3f dLocal(sinTheta * sinf(phi), cosTheta, -sinTheta * cosf(phi));

        // 4. Transform local direction to world space
        return normalize(toWorld * dLocal);
    }

    // Evaluate the solid angle PDF of the sampled direction.
    HD float pdf(const Vector3f& dirWorld) const {
        if (hasConstant) {
            return 1.f / (4.f * M_PI);
        }
        if (!hasMap || pixels == nullptr || width == 0 || height == 0 || marginalSum <= 0.f) {
            return 0.f;
        }

        // Convert world direction to local environment space
        const Vector3f d = normalize(Vector3f(
            toWorld.m[0][0] * dirWorld.x + toWorld.m[1][0] * dirWorld.y + toWorld.m[2][0] * dirWorld.z,
            toWorld.m[0][1] * dirWorld.x + toWorld.m[1][1] * dirWorld.y + toWorld.m[2][1] * dirWorld.z,
            toWorld.m[0][2] * dirWorld.x + toWorld.m[1][2] * dirWorld.y + toWorld.m[2][2] * dirWorld.z
        ));

        const float phi = atan2f(d.x, -d.z);
        float u = phi * INV_TWOPI;
        if (u < 0.f) u += 1.f;
        const float v = acosf(clamp(d.y, -1.f, 1.f)) * INV_PI;

        const int x = clamp((int)(u * width), 0, (int)width - 1);
        const int y = clamp((int)(v * height), 0, (int)height - 1);

        const Color3f color = pixels[y * width + x];
        const float lum = getLuminance(color);

        float sinThetaDir = sqrtf(fmaxf(0.f, 1.f - d.y * d.y));
        if (sinThetaDir <= 0.f) return 0.f;

        float sinThetaRow = sinf(M_PI * (y + 0.5f) / height);
        
        // Solid angle PDF = (continuous joint PDF) / (2 * PI * PI * sin(theta))
        return (lum * sinThetaRow) / (marginalSum * sinThetaDir) * (width * height) / (2.f * M_PI * M_PI);
    }

    HD Color3f sample(const Point2f& sample, Vector3f& dirWorld, float& outPdf) const {
        dirWorld = sampleDirection(sample);
        outPdf = pdf(dirWorld);
        return eval(dirWorld);
    }

    void setMap(const Color3f* hostPixels, uint32_t mapWidth, uint32_t mapHeight, const Matrix4f& envToWorld) {
        clear();

        width = mapWidth;
        height = mapHeight;
        toWorld = envToWorld;
        hasMap = (width > 0 && height > 0 && hostPixels != nullptr);
        hasConstant = false;
        constant = Color3f(0.f);

        if (!hasMap)
            return;

        // Allocate and copy pixels to GPU
        CUDA_CHECK(cudaMalloc(&pixels, width * height * sizeof(Color3f)));
        CUDA_CHECK(cudaMemcpy(pixels, hostPixels,
                                     width * height * sizeof(Color3f),
                                     cudaMemcpyHostToDevice));

        // 1. Build weights on host
        std::vector<float> weights(width * height);
        for (uint32_t y = 0; y < height; ++y) {
            float sinTheta = sinf(M_PI * (y + 0.5f) / height);
            for (uint32_t x = 0; x < width; ++x) {
                weights[y * width + x] = getLuminance(hostPixels[y * width + x]) * sinTheta;
            }
        }

        // 2. Construct 2D distribution on host
        Distribution2D dist;
        dist.build(weights, width, height);
        marginalSum = dist.marginal.funcSum;

        // 3. Flatten conditional CDFs for copying
        std::vector<float> condCdfsHost(height * (width + 1));
        for (uint32_t y = 0; y < height; ++y) {
            std::memcpy(&condCdfsHost[y * (width + 1)], dist.cond[y].cdf.data(), (width + 1) * sizeof(float));
        }

        // 4. Allocate and copy CDF arrays to GPU
        CUDA_CHECK(cudaMalloc(&condCdfs, condCdfsHost.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(condCdfs, condCdfsHost.data(),
                                     condCdfsHost.size() * sizeof(float),
                                     cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&marginalCdf, dist.marginal.cdf.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(marginalCdf, dist.marginal.cdf.data(),
                                     dist.marginal.cdf.size() * sizeof(float),
                                     cudaMemcpyHostToDevice));
    }

    void setConstant(const Color3f& radiance) {
        clear();
        width = 0;
        height = 0;
        hasMap = false;
        hasConstant = true;
        constant = radiance;
    }

    void clear() {
        if (pixels)      { CUDA_CHECK(cudaFree(pixels)); pixels = nullptr; }
        if (condCdfs)    { CUDA_CHECK(cudaFree(condCdfs)); condCdfs = nullptr; }
        if (marginalCdf) { CUDA_CHECK(cudaFree(marginalCdf)); marginalCdf = nullptr; }
        marginalSum = 0.f;

        width = 0;
        height = 0;
        hasMap = false;
        hasConstant = false;
        constant = Color3f(0.f);
    }
};

FUTABA_NAMESPACE_END
