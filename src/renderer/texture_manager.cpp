#include "texture_manager.h"
#include <iostream>
#include <cstring>
#include <stb_image.h>

FUTABA_NAMESPACE_BEGIN

cudaTextureObject_t TextureManager::createTexture(const std::string& filename, bool isSRGB) {
    int width = 0, height = 0, channels = 0;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 4); // load as RGBA
    stbi_set_flip_vertically_on_load(false);
    if (!data) {
        std::cerr << "Failed to load texture: " << filename << std::endl;
        return 0;
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray* cuArray = nullptr;
    cudaError_t err = cudaMallocArray(&cuArray, &channelDesc, width, height);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc array failed: " << cudaGetErrorString(err) << std::endl;
        stbi_image_free(data);
        return 0;
    }

    err = cudaMemcpy2DToArray(cuArray, 0, 0, data, width * 4 * sizeof(unsigned char), width * 4 * sizeof(unsigned char), height, cudaMemcpyHostToDevice);
    stbi_image_free(data);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy to array failed: " << cudaGetErrorString(err) << std::endl;
        cudaFreeArray(cuArray);
        return 0;
    }

    m_cudaTextureArrays.push_back(cuArray);

    struct cudaResourceDesc resDesc;
    std::memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    std::memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;
    texDesc.sRGB = isSRGB ? 1 : 0;

    cudaTextureObject_t texObj = 0;
    err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    if (err != cudaSuccess) {
        std::cerr << "CUDA create texture object failed: " << cudaGetErrorString(err) << std::endl;
        return 0;
    }

    m_cudaTextureObjects.push_back(texObj);
    return texObj;
}

void TextureManager::clear() {
    for (auto texObj : m_cudaTextureObjects) {
        if (texObj != 0) {
            cudaDestroyTextureObject(texObj);
        }
    }
    m_cudaTextureObjects.clear();

    for (auto cuArray : m_cudaTextureArrays) {
        if (cuArray != nullptr) {
            cudaFreeArray(cuArray);
        }
    }
    m_cudaTextureArrays.clear();
}

FUTABA_NAMESPACE_END
