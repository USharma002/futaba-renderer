#pragma once

#include "common.cuh"
#include "types.cuh"
#include <cmath>


namespace futaba {

struct Ray3f {
  Point3f o;     // Ray origin
  Vector3f d;    // Ray direction
  Vector3f dRcp; // Componentwise reciprocals of the ray direction
  float mint;    // Minimum position on the ray segment
  float maxt;    // Maximum position on the ray segment

  // Construct a new empty ray
  HD Ray3f() : mint(Epsilon), maxt(INFINITY) {}

  // Construct a new ray with origin and direction
  HD Ray3f(const Point3f &o, const Vector3f &d)
      : o(o), d(normalize(d)), mint(Epsilon), maxt(INFINITY) {
    update();
  }

  // Construct a new ray with custom segment bounds
  HD Ray3f(const Point3f &o, const Vector3f &d, float mint, float maxt)
      : o(o), d(normalize(d)), mint(mint), maxt(maxt) {
    update();
  }

  // Copy constructor
  HD Ray3f(const Ray3f &ray)
      : o(ray.o), d(ray.d), dRcp(ray.dRcp), mint(ray.mint), maxt(ray.maxt) {}

  // Copy a ray, but change the covered segment of the copy
  HD Ray3f(const Ray3f &ray, float mint, float maxt)
      : o(ray.o), d(ray.d), dRcp(ray.dRcp), mint(mint), maxt(maxt) {}

  // Update the reciprocal ray directions after changing 'd'
  HD void update() { dRcp = Vector3f(1.0f / d.x, 1.0f / d.y, 1.0f / d.z); }

  // Return the position of a point along the ray at distance 't'
  HD Point3f operator()(float t) const { return o + d * t; }

  // Return a ray that points into the opposite direction
  HD Ray3f reverse() const {
    Ray3f result;
    result.o = o;
    result.d = -d;
    result.dRcp = -dRcp;
    result.mint = mint;
    result.maxt = maxt;
    return result;
  }
};

typedef Ray3f Ray;

} // End namespace futaba