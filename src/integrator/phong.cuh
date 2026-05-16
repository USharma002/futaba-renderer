#pragma once

#include <cmath>

#include "common.cuh"
#include "sampler.cuh"
#include "scene.cuh"
#include "surface_interaction.cuh"


namespace futaba {

struct Phong {
  Vector3f lightDir;
  float ambientStrength;
  float diffuseStrength;
  float specularStrength;
  float shininess;

  HD Phong(const Vector3f &light_direction = Vector3f(1.0f, 1.0f, 1.0f),
           float ambient = 0.12f, float diffuse = 0.88f,
           float specular = 0.35f, float shiny = 32.0f)
      : lightDir(normalize(light_direction)), ambientStrength(ambient),
        diffuseStrength(diffuse), specularStrength(specular),
        shininess(shiny) {}

  HD Color3f sample(const Ray &ray, const Scene &scene, Sampler &) const {
    SurfaceIntersection si;
    if (!scene.intersect(ray, ray.mint, ray.maxt, si)) {
      return Color3f(0.0f);
    }

    Vector3f n = normalize(si.n);
    if (dot(ray.d, n) > 0.0f)
      n = -n;

    const Vector3f viewDir   = normalize(-ray.d);

    const Color3f lightColor(1.0f);

    const float ndotl = fmaxf(dot(n, lightDir), 0.0f);
    const Vector3f reflectDir = 2.0f * dot(n, lightDir) * n - lightDir;
    const float rdotv = fmaxf(dot(normalize(reflectDir), viewDir), 0.0f);

    Color3f diffuse  = si.albedo * (ambientStrength + diffuseStrength * ndotl);
    Color3f specular = si.specular * (specularStrength * powf(rdotv, shininess));

    return (diffuse + specular) * lightColor + si.emission;
  }
};

} // namespace futaba
