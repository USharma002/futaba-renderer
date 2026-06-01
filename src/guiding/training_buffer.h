#pragma once

#include "types.cuh"
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
    float* getActive() const { return m_dActive; }
    Point3f* getPosition() const { return m_dPosition; }
    Color3f* getNormals() const { return m_dNormals; }
    Color3f* getWi() const { return m_dWi; }
    Color3f* getWo() const { return m_dWo; }
    Color3f* getRadiance() const { return m_dRadiance; }
    float* getDirectionPdf() const { return m_dDirectionPdf; }
    float* getMaterialId() const { return m_dMaterialId; }

    // Run visualization kernel to write the selected channel/depth to a PBO
    void visualize(int width, int height, int maxDepth, int selectedDepth, int selectedBufferType, uchar4* d_pbo_ptr);

private:
    float* m_dActive = nullptr;
    Point3f* m_dPosition = nullptr;
    Color3f* m_dNormals = nullptr;
    Color3f* m_dWi = nullptr;
    Color3f* m_dWo = nullptr;
    Color3f* m_dRadiance = nullptr;
    float* m_dDirectionPdf = nullptr;
    float* m_dMaterialId = nullptr;

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
