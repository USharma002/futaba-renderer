#pragma once

#include "types.cuh"
#include "cuda_unique_ptr.h"
#include <vector_types.h>
#include <string>

namespace futaba {

struct TrainingBuffers {
    float* active = nullptr;
    Point3f* position = nullptr;
    Color3f* normals = nullptr;
    Color3f* wi = nullptr;
    Color3f* wo = nullptr;
    Color3f* radiance = nullptr;
    float* direction_pdf = nullptr;
    float* material_id = nullptr;
    int max_depth = 0;
    int pixel_index = -1;
    int img_size = 0;
};

class TrainingBufferManager {
public:
    TrainingBufferManager();
    ~TrainingBufferManager();

    // Allocate GPU buffers based on resolution and max depth
    void allocate(int width, int height, int maxDepth);

    // Clear all training buffers on the GPU to zero/default values
    void clear();

    // Free all allocated training buffers
    void freeBuffers();

    // Save training buffers to flat binary files with a companion metadata file
    void save(const std::string& basePath, int width, int height, int maxDepth);

    // Getters for device pointers
    float* getActive() const { return m_dActive.get(); }
    Point3f* getPosition() const { return m_dPosition.get(); }
    Color3f* getNormals() const { return m_dNormals.get(); }
    Color3f* getWi() const { return m_dWi.get(); }
    Color3f* getWo() const { return m_dWo.get(); }
    Color3f* getRadiance() const { return m_dRadiance.get(); }
    float* getDirectionPdf() const { return m_dDirectionPdf.get(); }
    float* getMaterialId() const { return m_dMaterialId.get(); }

    // Run visualization kernel to write the selected channel/depth to a PBO
    void visualize(int width, int height, int maxDepth, int selectedDepth, int selectedBufferType, uchar4* d_pbo_ptr);

private:
    CudaUniquePtr<float> m_dActive;
    CudaUniquePtr<Point3f> m_dPosition;
    CudaUniquePtr<Color3f> m_dNormals;
    CudaUniquePtr<Color3f> m_dWi;
    CudaUniquePtr<Color3f> m_dWo;
    CudaUniquePtr<Color3f> m_dRadiance;
    CudaUniquePtr<float> m_dDirectionPdf;
    CudaUniquePtr<float> m_dMaterialId;

    size_t m_allocatedCount = 0;
};

extern void run_visualization_kernel(
    const float* train_active,
    const Point3f* train_position,
    const Color3f* train_normals,
    const Color3f* train_wi,
    const Color3f* train_wo,
    const Color3f* train_radiance,
    const float* train_material_id,
    int width, int height,
    int max_depth,
    int selected_depth,
    int selected_buffer_type,
    uchar4* pbo_ptr
);

} // namespace futaba
