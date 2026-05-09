#include "common.cuh"
#include "types.cuh"

namespace futaba {

class Shape {
public:
  Color3f albedo;
  Color3f emission;

  HD Shape() : albedo(1.0f), emission(0.0f) {}
  HD Shape(const Color3f &albedo, const Color3f &emission)
      : albedo(albedo), emission(emission) {}

  // This is a pure virtual function that all derived shapes must implement
  HD virtual bool ray_intersect(const Ray3f &ray) const = 0;
};

} // namespace futaba