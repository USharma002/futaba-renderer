#include "training_buffer.h"
#include "common.cuh"
#include "tonemapping.cuh"
#include <fstream>
#include <vector>
#include <iostream>

namespace futaba {

// CUDA kernel to visualize the selected training buffer at a given depth
__global__ void visualize_training_buffer_kernel(
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
    uchar4* pbo_ptr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int buf_idx = selected_depth * (width * height) + idx;

    float active_val = train_active ? train_active[buf_idx] : 0.f;
    Color3f final_color(0.f);

    if (active_val > 0.5f) {
        if (selected_buffer_type == 0) {
            // Active
            final_color = Color3f(active_val);
        }
        else if (selected_buffer_type == 1) {
            // Position
            Point3f p = train_position ? train_position[buf_idx] : Point3f(0.f);
            final_color = tonemap::apply(Color3f(p.x, p.y, p.z) * 0.1f, 0);
        }
        else if (selected_buffer_type == 2) {
            // Normals
            Color3f n = train_normals ? train_normals[buf_idx] : Color3f(0.f);
            final_color = 0.5f * (safe_normalize(n) + Color3f(1.f));
        }
        else if (selected_buffer_type == 3) {
            // Incoming Angle wi
            Color3f wi = train_wi ? train_wi[buf_idx] : Color3f(0.f);
            final_color = 0.5f * (safe_normalize(wi) + Color3f(1.f));
        }
        else if (selected_buffer_type == 4) {
            // Outgoing Angle wo
            Color3f wo = train_wo ? train_wo[buf_idx] : Color3f(0.f);
            final_color = 0.5f * (safe_normalize(wo) + Color3f(1.f));
        }
        else if (selected_buffer_type == 5) {
            // Incoming Radiance
            Color3f val = train_radiance ? train_radiance[buf_idx] : Color3f(0.f);
            final_color = tonemap::apply(val, 0);
        }
        else if (selected_buffer_type == 6) {
            // Material ID
            float val = train_material_id ? train_material_id[buf_idx] : -1.f;
            int id = (int)roundf(val);
            if (id < 0) {
                final_color = Color3f(0.f);
            } else {
                float r = ((id * 17) % 255) / 255.f;
                float g = ((id * 59) % 255) / 255.f;
                float b = ((id * 113) % 255) / 255.f;
                final_color = Color3f(r, g, b);
            }
        }
    }

    final_color = toSRGB(final_color);

    pbo_ptr[idx].x = (unsigned char)clamp(final_color.x * 255.f, 0.f, 255.f);
    pbo_ptr[idx].y = (unsigned char)clamp(final_color.y * 255.f, 0.f, 255.f);
    pbo_ptr[idx].z = (unsigned char)clamp(final_color.z * 255.f, 0.f, 255.f);
    pbo_ptr[idx].w = 255;
}

TrainingBufferManager::TrainingBufferManager() {}

TrainingBufferManager::~TrainingBufferManager() {
    freeBuffers();
}

void TrainingBufferManager::allocate(int width, int height, int maxDepth) {
    freeBuffers();

    m_allocatedCount = (size_t)width * height * maxDepth;

    float* d_active = nullptr;
    Point3f* d_position = nullptr;
    Color3f* d_normals = nullptr;
    Color3f* d_wi = nullptr;
    Color3f* d_wo = nullptr;
    Color3f* d_radiance = nullptr;
    float* d_direction_pdf = nullptr;
    float* d_material_id = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_active), m_allocatedCount * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&d_position), m_allocatedCount * sizeof(Point3f));
    cudaMalloc(reinterpret_cast<void **>(&d_normals), m_allocatedCount * sizeof(Color3f));
    cudaMalloc(reinterpret_cast<void **>(&d_wi), m_allocatedCount * sizeof(Color3f));
    cudaMalloc(reinterpret_cast<void **>(&d_wo), m_allocatedCount * sizeof(Color3f));
    cudaMalloc(reinterpret_cast<void **>(&d_radiance), m_allocatedCount * sizeof(Color3f));
    cudaMalloc(reinterpret_cast<void **>(&d_direction_pdf), m_allocatedCount * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&d_material_id), m_allocatedCount * sizeof(float));

    m_dActive.reset(d_active);
    m_dPosition.reset(d_position);
    m_dNormals.reset(d_normals);
    m_dWi.reset(d_wi);
    m_dWo.reset(d_wo);
    m_dRadiance.reset(d_radiance);
    m_dDirectionPdf.reset(d_direction_pdf);
    m_dMaterialId.reset(d_material_id);

    clear();
}

void TrainingBufferManager::clear() {
    if (m_allocatedCount == 0) return;

    cudaMemset(m_dActive.get(), 0, m_allocatedCount * sizeof(float));
    cudaMemset(m_dPosition.get(), 0, m_allocatedCount * sizeof(Point3f));
    cudaMemset(m_dNormals.get(), 0, m_allocatedCount * sizeof(Color3f));
    cudaMemset(m_dWi.get(), 0, m_allocatedCount * sizeof(Color3f));
    cudaMemset(m_dWo.get(), 0, m_allocatedCount * sizeof(Color3f));
    cudaMemset(m_dRadiance.get(), 0, m_allocatedCount * sizeof(Color3f));
    cudaMemset(m_dDirectionPdf.get(), 0, m_allocatedCount * sizeof(float));
    cudaMemset(m_dMaterialId.get(), 0, m_allocatedCount * sizeof(float));
}

void TrainingBufferManager::freeBuffers() {
    m_dActive.reset();
    m_dPosition.reset();
    m_dNormals.reset();
    m_dWi.reset();
    m_dWo.reset();
    m_dRadiance.reset();
    m_dDirectionPdf.reset();
    m_dMaterialId.reset();

    m_allocatedCount = 0;
}

void TrainingBufferManager::save(const std::string& basePath, int width, int height, int maxDepth) {
    if (m_allocatedCount == 0) return;

    std::string prefix = basePath;
    if (prefix.size() > 4 && prefix.substr(prefix.size() - 4) == ".bin") {
        prefix = prefix.substr(0, prefix.size() - 4);
    }

    size_t count = m_allocatedCount;

    // 1. Active
    if (m_dActive) {
        std::vector<float> host(count);
        cudaMemcpy(host.data(), m_dActive.get(), count * sizeof(float), cudaMemcpyDeviceToHost);
        std::ofstream out(prefix + "_active.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(host.data()), count * sizeof(float));
    }

    // 2. Position
    if (m_dPosition) {
        std::vector<Point3f> host(count);
        cudaMemcpy(host.data(), m_dPosition.get(), count * sizeof(Point3f), cudaMemcpyDeviceToHost);
        std::ofstream out(prefix + "_position.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(host.data()), count * sizeof(Point3f));
    }

    // 3. Normals
    if (m_dNormals) {
        std::vector<Color3f> host(count);
        cudaMemcpy(host.data(), m_dNormals.get(), count * sizeof(Color3f), cudaMemcpyDeviceToHost);
        std::ofstream out(prefix + "_normals.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(host.data()), count * sizeof(Color3f));
    }

    // 4. Incoming Angle (wi)
    if (m_dWi) {
        std::vector<Color3f> host(count);
        cudaMemcpy(host.data(), m_dWi.get(), count * sizeof(Color3f), cudaMemcpyDeviceToHost);
        std::ofstream out(prefix + "_wi.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(host.data()), count * sizeof(Color3f));
    }

    // 5. Outgoing Angle (wo)
    if (m_dWo) {
        std::vector<Color3f> host(count);
        cudaMemcpy(host.data(), m_dWo.get(), count * sizeof(Color3f), cudaMemcpyDeviceToHost);
        std::ofstream out(prefix + "_wo.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(host.data()), count * sizeof(Color3f));
    }

    // 6. Incoming Radiance
    if (m_dRadiance) {
        std::vector<Color3f> host(count);
        cudaMemcpy(host.data(), m_dRadiance.get(), count * sizeof(Color3f), cudaMemcpyDeviceToHost);
        std::ofstream out(prefix + "_radiance.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(host.data()), count * sizeof(Color3f));
    }

    // 7. Direction PDF
    if (m_dDirectionPdf) {
        std::vector<float> host(count);
        cudaMemcpy(host.data(), m_dDirectionPdf.get(), count * sizeof(float), cudaMemcpyDeviceToHost);
        std::ofstream out(prefix + "_direction_pdf.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(host.data()), count * sizeof(float));
    }

    // 8. Material ID
    if (m_dMaterialId) {
        std::vector<float> host(count);
        cudaMemcpy(host.data(), m_dMaterialId.get(), count * sizeof(float), cudaMemcpyDeviceToHost);
        std::ofstream out(prefix + "_material_id.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(host.data()), count * sizeof(float));
    }

    // Metadata
    std::ofstream meta(prefix + "_metadata.txt");
    meta << "width: " << width << "\n";
    meta << "height: " << height << "\n";
    meta << "depth: " << maxDepth << "\n";
    meta << "channels_active: 1\n";
    meta << "channels_position: 3\n";
    meta << "channels_normals: 3\n";
    meta << "channels_wi: 3\n";
    meta << "channels_wo: 3\n";
    meta << "channels_radiance: 3\n";
    meta << "channels_direction_pdf: 1\n";
    meta << "channels_material_id: 1\n";
    meta.close();

    std::cout << "[Dataset] Saved training dataset with prefix: " << prefix << std::endl;
}

void TrainingBufferManager::visualize(
    int width, int height, int maxDepth,
    int selectedDepth, int selectedBufferType,
    uchar4* d_pbo_ptr)
{
    if (!d_pbo_ptr) return;

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    visualize_training_buffer_kernel<<<grid, block>>>(
        m_dActive.get(),
        m_dPosition.get(),
        m_dNormals.get(),
        m_dWi.get(),
        m_dWo.get(),
        m_dRadiance.get(),
        m_dMaterialId.get(),
        width, height,
        maxDepth,
        selectedDepth,
        selectedBufferType,
        d_pbo_ptr
    );
    cudaDeviceSynchronize();
}

void run_visualization_kernel(
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
    uchar4* pbo_ptr)
{
    if (!pbo_ptr) return;
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    visualize_training_buffer_kernel<<<grid, block>>>(
        train_active,
        train_position,
        train_normals,
        train_wi,
        train_wo,
        train_radiance,
        train_material_id,
        width, height,
        max_depth,
        selected_depth,
        selected_buffer_type,
        pbo_ptr
    );
}

} // namespace futaba
