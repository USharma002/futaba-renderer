#pragma once

#include "types.cuh"
#include "cuda_unique_ptr.h"
#include <cuda_runtime.h>

namespace futaba {

// GPU scratch buffers filled by the path integrator during a training pass.
// Layout is one entry per (pixel, path vertex): slot = pixel * maxDepth + depth.
// The host reads these back and builds the SD-tree on the CPU.
class TrainingBufferManager {
public:
    void allocate(int width, int height, int maxDepth) {
        const size_t count = static_cast<size_t>(width) * height * maxDepth;
        if (count == m_count) return;
        m_count = count;
        m_active.reset(alloc<float>(count));
        m_position.reset(alloc<Point3f>(count));
        m_wo.reset(alloc<Vector3f>(count));
        m_radiance.reset(alloc<Color3f>(count));
        m_pdf.reset(alloc<float>(count));
    }

    // Only 'active' needs zeroing: the host skips inactive slots.
    void clear() {
        if (m_count > 0) cudaMemset(m_active.get(), 0, m_count * sizeof(float));
    }

    size_t    count()    const { return m_count; }
    float*    active()   const { return m_active.get(); }
    Point3f*  position() const { return m_position.get(); }
    Vector3f* wo()       const { return m_wo.get(); }
    Color3f*  radiance() const { return m_radiance.get(); }
    float*    pdf()      const { return m_pdf.get(); }

private:
    template <typename T>
    static T* alloc(size_t count) {
        T* ptr = nullptr;
        cudaMalloc(&ptr, count * sizeof(T));
        return ptr;
    }

    CudaUniquePtr<float>    m_active;
    CudaUniquePtr<Point3f>  m_position;
    CudaUniquePtr<Vector3f> m_wo;
    CudaUniquePtr<Color3f>  m_radiance;
    CudaUniquePtr<float>    m_pdf;
    size_t m_count = 0;
};

} // namespace futaba
