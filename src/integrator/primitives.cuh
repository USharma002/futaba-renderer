#pragma once

#include "sampler.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"


namespace futaba {

HD Color3f faceColorFromId(int primitiveId) {
  unsigned int x = (unsigned int)(primitiveId >= 0 ? primitiveId : 0);
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;

  unsigned int y = x * 1664525u + 1013904223u;
  unsigned int z = y * 1664525u + 1013904223u;

  auto toChannel = [](unsigned int v) -> float {
    return 0.35f + 0.65f * ((v & 0x00ffffffu) / 16777215.0f);
  };

  return Color3f(toChannel(x), toChannel(y), toChannel(z));
}

struct Primitives {
  // Visualizes each face with a stable pseudo-random color for debugging.
  HD Color3f sample(const Ray &ray, const Scene &scene, Sampler &) const {
    SurfaceIntersection si;
    if (!scene.intersect(ray, ray.mint, ray.maxt, si)) {
      return Color3f(0.0f);
    }

    int faceId = si.primitive_id;
    if (faceId < 0)
      faceId = si.shape_id;

    return faceColorFromId(faceId);
  }
};
} // namespace futaba
