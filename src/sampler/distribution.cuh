#pragma once

#include <vector>
#include <numeric>
#include <cmath>
#include <cuda_runtime.h>

#include "types.cuh"

namespace futaba {

HD void sample1D_device(const float* cdf, int n, float funcSum, float u, int &index, float &du, float &pdf);

struct Distribution1D {
    std::vector<float> func;
    std::vector<float> cdf;
    float funcSum = 0.f;

    Distribution1D() = default;

    void build(const std::vector<float>& f) {
        func = f;
        const int n = (int)func.size();

        cdf.resize(n + 1);
        cdf[0] = 0.f;
        for (int i = 0; i < n; ++i) {
            cdf[i+1] = cdf[i] + std::max(0.f, func[i]);
        }
        funcSum = cdf[n];
        if (funcSum > 0.f) {
            for (int i = 1; i <= n; ++i) cdf[i] /= funcSum;
        } else {
            for (int i = 1; i <= n; ++i) cdf[i] = float(i) / float(n);
            funcSum = 0.f;
        }
    }

    void sample(float u, int& index, float& pdf, float& du) const {
        sample1D_device(cdfData(), size(), funcSum, u, index, du, pdf);
    }
    const float* cdfData() const { return cdf.empty() ? nullptr : cdf.data(); }
    int size() const { return (int)func.size(); }
};

struct Distribution2D {
    Distribution1D marginal;
    std::vector<Distribution1D> cond;
    int nu = 0, nv = 0;

    Distribution2D() = default;

    void build(const std::vector<float>& f, int _nu, int _nv) {
        nu = _nu; nv = _nv;
        cond.resize(nv);
        std::vector<float> rowSums(nv, 0.f);
        for (int v = 0; v < nv; ++v) {
            std::vector<float> row(nu);
            for (int u = 0; u < nu; ++u) {
                row[u] = f[v * nu + u];
            }
            cond[v].build(row);
            rowSums[v] = cond[v].funcSum;
        }
        marginal.build(rowSums);
    }
};

HD int findIntervalDevice(const float* cdf, int n, float u) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cdf[mid+1] <= u) lo = mid + 1; else hi = mid;
    }
    return lo;
}

HD void sample1D_device(const float* cdf, int n, float funcSum, float u, int &index, float &du, float &pdf) {
    if (n <= 0) { index = -1; du = 0.f; pdf = 0.f; return; }
    index = findIntervalDevice(cdf, n, u);
    float c0 = cdf[index];
    float c1 = cdf[index+1];
    if (c1 - c0 <= 0.f) {
        du = 0.f;
        pdf = (funcSum > 0.f) ? (1.f / float(n)) : 0.f;
    } else {
        du = (u - c0) / (c1 - c0);
        pdf = (funcSum > 0.f) ? ((c1 - c0) * funcSum) : 0.f;
    }
}

HD void sample2D_device(const float* condCdfs, int nu, int nv, const float* rowSums, const float* marginalCdf, float marginalSum, const Point2f& samp, int &iu, int &iv, float &du, float &dv, float &pdf) {
    float pdf_v = 0.f;
    sample1D_device(marginalCdf, nv, marginalSum, samp.y, iv, dv, pdf_v);

    const float* rowCdf = condCdfs + (size_t)iv * (nu + 1);
    float pdf_u = 0.f;
    sample1D_device(rowCdf, nu, rowSums[iv], samp.x, iu, du, pdf_u);

    pdf = pdf_v * pdf_u;
}

} // namespace futaba
