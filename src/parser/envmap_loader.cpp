#include "envmap_loader.h"
#include <fstream>
#include <filesystem>
#include <stb_image.h>

#include <tinyexr.h>
#include <stb_image_write.h>



namespace fs = std::filesystem;

FUTABA_NAMESPACE_BEGIN

bool loadEnvMapEXR(const std::string& filename,
                   const std::string& baseDir,
                   std::vector<Color3f>& pixels,
                   int& width,
                   int& height,
                   std::string& errorOut)
{
    fs::path path = fs::path(baseDir) / filename;
    float* rgba = nullptr;
    int w = 0, h = 0;
    const char* err = nullptr;

    int ret = LoadEXR(&rgba, &w, &h, path.string().c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        errorOut = "Failed to load envmap EXR: " + path.string();
        if (err) {
            errorOut += " - " + std::string(err);
            FreeEXRErrorMessage(err);
        }
        return false;
    }

    pixels.resize((size_t)w * (size_t)h);
    for (int i = 0; i < w * h; ++i) {
        pixels[i] = Color3f(rgba[4 * i + 0], rgba[4 * i + 1], rgba[4 * i + 2]);
    }
    free(rgba);
    width = w;
    height = h;
    return true;
}

bool loadEnvMapHDR(const std::string& filename,
                   const std::string& baseDir,
                   std::vector<Color3f>& pixels,
                   int& width,
                   int& height,
                   std::string& errorOut)
{
    fs::path path = fs::path(baseDir) / filename;
    std::ifstream f(path, std::ios::binary);
    bool isEXR = false;
    if (f.is_open()) {
        char magic[4] = {0};
        if (f.read(magic, 4)) {
            if (magic[0] == 0x76 && magic[1] == 0x2f && magic[2] == 0x31 && magic[3] == 0x01) {
                isEXR = true;
            }
        }
        f.close();
    } else {
        errorOut = "Cannot open environment map file: " + path.string();
        return false;
    }

    if (isEXR) {
        return loadEnvMapEXR(filename, baseDir, pixels, width, height, errorOut);
    }

    int w = 0, h = 0, comp = 0;
    float* data = stbi_loadf(path.string().c_str(), &w, &h, &comp, 3);
    if (!data) {
        errorOut = "Failed to load envmap HDR: " + path.string();
        return false;
    }

    pixels.resize((size_t)w * (size_t)h);
    for (int i = 0; i < w * h; ++i) {
        pixels[i] = Color3f(data[3 * i + 0], data[3 * i + 1], data[3 * i + 2]);
    }
    stbi_image_free(data);
    width = w;
    height = h;
    return true;
}

FUTABA_NAMESPACE_END
