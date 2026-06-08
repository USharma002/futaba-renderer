#pragma once

#include <vector>
#include <string>
#include <cuda_runtime.h>

namespace futaba {

class TextureManager {
public:
    TextureManager() = default;
    ~TextureManager() { clear(); }

    // Disable copy/assignment
    TextureManager(const TextureManager&) = delete;
    TextureManager& operator=(const TextureManager&) = delete;

    cudaTextureObject_t createTexture(const std::string& filename);
    void clear();

    const std::vector<cudaArray*>& getArrays() const { return m_cudaTextureArrays; }
    const std::vector<cudaTextureObject_t>& getTextures() const { return m_cudaTextureObjects; }

private:
    std::vector<cudaArray*>          m_cudaTextureArrays;
    std::vector<cudaTextureObject_t> m_cudaTextureObjects;
};

} // namespace futaba
