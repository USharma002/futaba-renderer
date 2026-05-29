#pragma once

#include <vector>
#include <numeric>
#include <cmath>
#include <cuda_runtime.h>

#include "types.cuh"

namespace futaba {

HD void sample1D_device(const float* cdf, int n, float funcSum, float u, int &index, float &du, float &pdf);

// Simple 1D distribution helper (host-side construction, device-side sampling
// helpers are provided as HD functions). This is intentionally minimal and
// designed to be embedded into device-visible arrays by copying raw floats.
struct Distribution1D {
    // Host-side arrays
    std::vector<float> func;    // input function values
    std::vector<float> cdf;     // cumulative distribution (size = n+1)
    float funcSum = 0.f;

    Distribution1D() = default;

    // Build distribution from function values (non-negative)
    void build(const std::vector<float>& f) {
        func = f;
        const int n = (int)func.size();

        // Build a CDF using exclusive prefix sum. cdf[0] = 0, cdf[n] = funcSum.
        
        cdf.resize(n + 1);
        cdf[0] = 0.f;
        for (int i = 0; i < n; ++i) {
            cdf[i+1] = cdf[i] + std::max(0.f, func[i]);
        }
        funcSum = cdf[n];
        
        // if funcSum is zero (all zero function), make a uniform distribution to avoid NaNs in sampling.
        if (funcSum > 0.f) {
            for (int i = 1; i <= n; ++i) cdf[i] /= funcSum;
        } else {
            // Make uniform if all zero
            for (int i = 1; i <= n; ++i) cdf[i] = float(i) / float(n);
            funcSum = 0.f;
        }
    }

    // Sample discrete index and return pdf and offset within cell [0,1)
    // u in [0,1)
    void sample(float u, int& index, float& pdf, float& du) const {
        sample1D_device(cdfData(), size(), funcSum, u, index, du, pdf);
    }

    // Host accessor to raw arrays for transfer
    const float* cdfData() const { return cdf.empty() ? nullptr : cdf.data(); }
    int size() const { return (int)func.size(); }
};

// 2D distribution: marginal over rows and conditional distributions per row.
struct Distribution2D {
    Distribution1D marginal;               // marginal over v (rows)
    std::vector<Distribution1D> cond;     // conditional CDFs for each row (u)
    int nu = 0, nv = 0;

    Distribution2D() = default;

    // Build from a flattened function values of size nu*nv (row-major: v rows each with nu samples)
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

// Device helpers: binary search on device CDF (cdf array length = n+1)
HD int findIntervalDevice(const float* cdf, int n, float u) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cdf[mid+1] <= u) lo = mid + 1; else hi = mid;
    }
    return lo;
}

// Sample 1D device-side given cdf (size n+1), returns index and du in [0,1), and pdf in func-sum units
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

// Sample 2D device-side. Arguments:
// - condCdfs: pointer to array of nv pointers (flattened) each pointing to a cdf of length nu+1
// - marginalCdf: pointer to marginal cdf length nv+1
// - funcSums: pointer to nv funcSum values per row
// Note: For simplicity we expect layout where condCdfs is a contiguous block of (nv*(nu+1)) floats.
HD void sample2D_device(const float* condCdfs, int nu, int nv, const float* rowSums, const float* marginalCdf, float marginalSum, const Point2f& samp, int &iu, int &iv, float &du, float &dv, float &pdf) {
    float pdf_v = 0.f;
    sample1D_device(marginalCdf, nv, marginalSum, samp.y, iv, dv, pdf_v);

    const float* rowCdf = condCdfs + (size_t)iv * (nu + 1);
    float pdf_u = 0.f;
    sample1D_device(rowCdf, nu, rowSums[iv], samp.x, iu, du, pdf_u);

    pdf = pdf_v * pdf_u; // joint pdf in original func units (not normalized to 1)
}

} // namespace futaba
