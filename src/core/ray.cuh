#pragma once

#include "common.cuh"
#include "types.cuh"
#include <cmath>


FUTABA_NAMESPACE_BEGIN

struct Ray3f {
  Point3f o;     // Ray origin (12 bytes)
  Vector3f d;    // Ray direction (12 bytes)
  float mint;    // Minimum position on the ray segment (4 bytes)
  float maxt;    // Maximum position on the ray segment (4 bytes)

  // Construct a new empty ray
  HD Ray3f() : mint(Epsilon), maxt(Infinity) {}

  // Construct a new ray with origin and direction
  HD Ray3f(const Point3f &o, const Vector3f &d)
      : o(o), d(normalize(d)), mint(Epsilon), maxt(Infinity) {}

  // Construct a new ray with custom segment bounds
  HD Ray3f(const Point3f &o, const Vector3f &d, float mint, float maxt)
      : o(o), d(normalize(d)), mint(mint), maxt(maxt) {}

  // Copy constructor
  HD Ray3f(const Ray3f &ray)
      : o(ray.o), d(ray.d), mint(ray.mint), maxt(ray.maxt) {}

  // Copy a ray, but change the covered segment of the copy
  HD Ray3f(const Ray3f &ray, float mint, float maxt)
      : o(ray.o), d(ray.d), mint(mint), maxt(maxt) {}

  // Return the position of a point along the ray at distance 't'
  HD Point3f operator()(float t) const { return o + d * t; }

  // Return a ray that points into the opposite direction
  HD Ray3f reverse() const {
    Ray3f result;
    result.o = o;
    result.d = -d;
    result.mint = mint;
    result.maxt = maxt;
    return result;
  }
};

typedef Ray3f Ray;

FUTABA_NAMESPACE_END