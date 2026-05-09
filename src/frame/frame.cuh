#pragma once

#include "common.cuh"

namespace futaba {

struct Frame {
  Vector3f s;
  Vector3f t;
  Vector3f n;

  HD Frame() : s(1.0f, 0.0f, 0.0f), t(0.0f, 1.0f, 0.0f), n(0.0f, 0.0f, 1.0f) {}
  HD explicit Frame(const Vector3f &normal) { setFromNormal(normal); }

  HD void setFromNormal(const Vector3f &normal) {
    n = normalize(normal);
    float sign = (n.z >= 0.0f) ? 1.0f : -1.0f;
    float a = -1.0f / (sign + n.z);
    float b = n.x * n.y * a;

    s = Vector3f(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    t = Vector3f(b, sign + n.y * n.y * a, -n.y);
  }

  HD static float cos_theta(const Vector3f &v) { return v.z; }

  HD static float absCosTheta(const Vector3f &v) { return fabsf(v.z); }

  HD Vector3f to_local(const Vector3f &v) const {
    return Vector3f(dot(v, s), dot(v, t), dot(v, n));
  }

  HD Vector3f to_world(const Vector3f &v) const {
    return s * v.x + t * v.y + n * v.z;
  }
};

} // namespace futaba
