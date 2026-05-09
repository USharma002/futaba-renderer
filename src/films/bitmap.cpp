#include "bitmap.h"
#include "hdrfilm.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

// TinyEXR — single-header EXR writer (public domain)
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

// STB Image Write for zlib compression
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace futaba {

Bitmap* HDRFilm::toBitmap() const {
    Bitmap* bmp = new Bitmap(width, height);
    
    // Copy pixels from GPU to CPU
    cudaMemcpy(bmp->pixels.data(), d_pixels, width * height * sizeof(Color3f), cudaMemcpyDeviceToHost);
    
    // Divide by sampleCount to get average radiance
    if (sampleCount > 1) {
        float invSampleCount = 1.0f / (float)sampleCount;
        for (int i = 0; i < width * height; ++i) {
            bmp->pixels[i] *= invSampleCount;
        }
    }
    
    return bmp;
}

bool Bitmap::saveEXR(const std::string& filename) const {
    if (width <= 0 || height <= 0 || pixels.empty()) {
        std::cerr << "Bitmap::saveEXR(): empty image, nothing to save." << std::endl;
        return false;
    }

    // Append .exr extension if not already present
    std::string path = filename;
    if (path.size() < 4 || path.substr(path.size() - 4) != ".exr") {
        path += ".exr";
    }

    std::cout << "Writing a " << width << "x" << height
              << " OpenEXR file to \"" << path << "\"" << std::endl;

    // De-interleave RGB into separate channels (TinyEXR wants planar data)
    std::vector<float> r(width * height);
    std::vector<float> g(width * height);
    std::vector<float> b(width * height);

    for (int i = 0; i < width * height; ++i) {
        r[i] = pixels[i].x;
        g[i] = pixels[i].y;
        b[i] = pixels[i].z;
    }

    // TinyEXR expects channels in alphabetical order: B, G, R
    float* channels[] = { b.data(), g.data(), r.data() };

    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 3;
    image.images = reinterpret_cast<unsigned char**>(channels);
    image.width = width;
    image.height = height;

    header.num_channels = 3;

    EXRChannelInfo channel_infos[3];
    header.channels = channel_infos;

    // Channels must be in alphabetical order for EXR
    strncpy(channel_infos[0].name, "B", 255); channel_infos[0].name[1] = '\0';
    strncpy(channel_infos[1].name, "G", 255); channel_infos[1].name[1] = '\0';
    strncpy(channel_infos[2].name, "R", 255); channel_infos[2].name[1] = '\0';

    int pixel_types[] = { TINYEXR_PIXELTYPE_FLOAT, TINYEXR_PIXELTYPE_FLOAT, TINYEXR_PIXELTYPE_FLOAT };
    int requested_pixel_types[] = { TINYEXR_PIXELTYPE_HALF, TINYEXR_PIXELTYPE_HALF, TINYEXR_PIXELTYPE_HALF };
    header.pixel_types = pixel_types;
    header.requested_pixel_types = requested_pixel_types;

    const char* err = nullptr;
    int ret = SaveEXRImageToFile(&image, &header, path.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        std::cerr << "Bitmap::saveEXR(): failed to write \"" << path << "\"";
        if (err) {
            std::cerr << " — " << err;
            FreeEXRErrorMessage(err);
        }
        std::cerr << std::endl;
        return false;
    }

    return true;
}

} // namespace futaba
