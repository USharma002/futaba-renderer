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

#if !defined(__CUDACC__) && defined(NANOGUI_GLAD)
#include "integrator_ui.h"
#include <nanogui/nanogui.h>

namespace futaba {

class PhongIntegratorUI : public IntegratorUI {
public:
    PhongIntegratorUI()
        : m_lightDir(1.f, 1.f, 1.f),
          m_ambient(0.12f),
          m_diffuse(0.88f),
          m_specular(0.35f),
          m_shininess(32.f) {}

    std::string getName() const override { return "Phong"; }
    int getMode() const override { return INTEGRATOR_PHONG; }

    void renderUI(nanogui::Widget* parent, std::function<void()> clearFilm) override {
        auto addPhongSlider = [parent, clearFilm](const std::string &name, float minVal,
                                     float maxVal, float initialValue,
                                     const std::function<void(float)> &onChange) {
            new nanogui::Label(parent, name, "sans-bold");
            nanogui::Widget *panel = new nanogui::Widget(parent);
            panel->setLayout(new nanogui::BoxLayout(nanogui::Orientation::Horizontal, nanogui::Alignment::Middle, 0, 10));

            nanogui::Slider *slider = new nanogui::Slider(panel);
            slider->setFixedWidth(120);
            
            float t = (initialValue - minVal) / (maxVal - minVal);
            slider->setValue(t < 0.f ? 0.f : (t > 1.f ? 1.f : t));

            nanogui::Label *valueLabel = new nanogui::Label(panel, std::to_string(initialValue));
            slider->setCallback([minVal, maxVal, valueLabel, onChange, clearFilm](float t) {
                const float value = minVal + t * (maxVal - minVal);
                valueLabel->setCaption(std::to_string(value));
                onChange(value);
                clearFilm();
            });
        };

        addPhongSlider("Ambient", 0.f, 2.f, m_ambient, [this](float v) { m_ambient = v; });
        addPhongSlider("Diffuse", 0.f, 2.f, m_diffuse, [this](float v) { m_diffuse = v; });
        addPhongSlider("Specular", 0.f, 2.f, m_specular, [this](float v) { m_specular = v; });
        addPhongSlider("Shininess", 1.f, 128.f, m_shininess, [this](float v) { m_shininess = v; });
    }

    void updateLaunchParams(LaunchParams& params) const override {
        params.phong_light_dir = m_lightDir;
        params.phong_ambient = m_ambient;
        params.phong_diffuse = m_diffuse;
        params.phong_specular = m_specular;
        params.phong_shininess = m_shininess;
    }

private:
    Vector3f m_lightDir;
    float m_ambient;
    float m_diffuse;
    float m_specular;
    float m_shininess;
};

} // namespace futaba
#endif

