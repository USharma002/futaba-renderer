#pragma once

#include "types.cuh"
#include <string>
#include <vector>

namespace futaba {

// Bitmap: Image I/O and processing
class Bitmap {
public:
  int width, height;
  std::vector<Color3f> data;

  Bitmap(int w, int h) : width(w), height(h), data(w * h, Color3f(0.f)) {}

  // Write to file (PNG, EXR, etc.)
  bool save(const std::string &filename) const;

  // TODO: Implement EXR saving with OpenEXR library
  // bool saveEXR(const std::string& filename) const { ... }

  // TODO: Implement PNG saving
  // bool savePNG(const std::string& filename) const { ... }

  // Tone mapping: Convert HDR to LDR
  void tonemap(float exposure = 1.0f, float gamma = 2.2f);

  // Get pixel at (x, y)
  Color3f &at(int x, int y) { return data[y * width + x]; }

  const Color3f &at(int x, int y) const { return data[y * width + x]; }
};

// Inline implementations
inline bool Bitmap::save(const std::string &filename) const {
  // TODO: Implement file I/O
  // Check extension and dispatch to appropriate save function
  return false;
}

inline void Bitmap::tonemap(float exposure, float gamma) {
  // TODO: Implement tone mapping
  // for (auto& pixel : data) {
  //     pixel *= exposure;
  //     pixel = Color3f(powf(pixel.x, 1.f/gamma),
  //                     powf(pixel.y, 1.f/gamma),
  //                     powf(pixel.z, 1.f/gamma));
  // }
}

} // namespace futaba
