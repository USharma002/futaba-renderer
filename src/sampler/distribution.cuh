#pragma once

#include <vector>
#include <numeric>
#include <cmath>
#include <cuda_runtime.h>

#include "types.cuh"

namespace futaba {

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
        const int n = (int)func.size();
        if (n == 0) { index = -1; pdf = 0.f; du = 0.f; return; }

        // Binary search in cdf (host-side; device helper provided separately)
        int lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (cdf[mid+1] <= u) lo = mid + 1; else hi = mid;
        }
        index = lo;
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
    // sample row v
    float u1 = samp.y;
    int v = findIntervalDevice(marginalCdf, nv, u1);
    float mv0 = marginalCdf[v];
    float mv1 = marginalCdf[v+1];
    float dv_local = 0.f;
    float pdf_v = 0.f;
    if (mv1 - mv0 <= 0.f) {
        dv_local = 0.f;
        pdf_v = (marginalSum > 0.f) ? (1.f / float(nv)) : 0.f;
    } else {
        dv_local = (u1 - mv0) / (mv1 - mv0);
        pdf_v = (marginalSum > 0.f) ? ((mv1 - mv0) * marginalSum) : 0.f;
    }

    // sample column u within row v
    const float* rowCdf = condCdfs + (size_t)v * (nu + 1);
    float u0 = samp.x;
    int u = findIntervalDevice(rowCdf, nu, u0);
    float r0 = rowCdf[u];
    float r1 = rowCdf[u+1];
    float du_local = 0.f;
    float pdf_u = 0.f;
    if (r1 - r0 <= 0.f) {
        du_local = 0.f;
        pdf_u = (rowSums[v] > 0.f) ? (1.f / float(nu)) : 0.f;
    } else {
        du_local = (u0 - r0) / (r1 - r0);
        pdf_u = (rowSums[v] > 0.f) ? ((r1 - r0) * rowSums[v]) : 0.f;
    }

    iu = u; iv = v;
    du = du_local; dv = dv_local;
    pdf = pdf_v * pdf_u; // joint pdf in original func units (not normalized to 1)
}

} // namespace futaba
