#pragma once

#include "types.cuh"
#include <string>
#include <vector>

namespace futaba {

class Bitmap {
public:
  int width, height;
  std::vector<Color3f> pixels; // Row-major: pixel(x,y) = pixels[y * width + x]

  Bitmap() : width(0), height(0) {}
  Bitmap(int w, int h) : width(w), height(h), pixels(w * h, Color3f(0.f)) {}

  /// Access pixel at (x, y)
  Color3f &operator()(int x, int y) { return pixels[y * width + x]; }
  const Color3f &operator()(int x, int y) const {
    return pixels[y * width + x];
  }

  /// Save as OpenEXR file (HDR, linear). Appends ".exr" if not present.
  bool saveEXR(const std::string &filename) const;
};

} // namespace futaba
