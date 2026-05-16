#pragma once

#include <nanogui/nanogui.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "perspective.cuh"
#include "renderer.h"
#include "scene.cuh"
#include "scene_loader.h"

namespace futaba {
class HDRFilm;
}

class FutabaScreen : public nanogui::Screen {
public:
  FutabaScreen(int width, int height);
  ~FutabaScreen();
  void renderLoop();

protected:
  virtual bool keyboardEvent(int key, int scancode, int action,
                             int modifiers) override;
  virtual bool mouseButtonEvent(const nanogui::Vector2i &p, int button,
                                bool down, int modifiers) override;
  virtual bool mouseMotionEvent(const nanogui::Vector2i &p,
                                const nanogui::Vector2i &rel, int button,
                                int modifiers) override;
  virtual bool scrollEvent(const nanogui::Vector2i &p,
                           const nanogui::Vector2f &rel) override;
  virtual bool resizeEvent(const nanogui::Vector2i &size) override;

  void updateCamera();

  // Load a scene XML; returns false and shows a message box on failure.
  bool loadScene(const std::string &xmlPath);
  void recreateRenderTargets(int width, int height);
  void drawGizmo();

private:
  int m_renderWidth;
  int m_renderHeight;

  nanogui::Label *m_fpsLabel = nullptr;
  nanogui::Label *m_sceneLabel = nullptr;
  nanogui::Label *m_triCountLabel = nullptr;
  nanogui::Window *m_phongWindow = nullptr;
  nanogui::Window *m_settingsWindow = nullptr;
  nanogui::Slider *m_fovSlider = nullptr;
  nanogui::Slider *m_focusSlider = nullptr;
  nanogui::Slider *m_apertureSlider = nullptr;
  int m_maxDepth = 12;
  int m_rrDepth = 5;
  int m_integratorMode = futaba::INTEGRATOR_PATH;
  int m_tonemappingMode = futaba::TONEMAPPING_NONE;
  bool m_useVertexNormals = true;
  bool m_useAntialiasing = true;

  GLuint m_glTex = 0;
  GLuint m_glPbo = 0;
  GLuint m_fbo = 0;
  cudaGraphicsResource_t m_cudaPboResource = nullptr;

  futaba::HDRFilm *m_film = nullptr;
  futaba::PerspectiveCamera m_camera;
  futaba::Scene m_scene; // lives here; uploaded once per load

  // Input state
  bool m_keys[1024] = {false};
  bool m_rightMousePressed = false;

  // Camera state
  ::Vector3f m_camPos = ::Vector3f(0.f, 0.f, 2.5f);
  ::Vector3f m_camForward = ::Vector3f(0.f, 0.f, -1.f);
  ::Vector3f m_camUp = ::Vector3f(0.f, 1.f, 0.f);
  float m_moveSpeed = 2.f;
  float m_currentFov = 39.3077f;
  float m_currentFocusDistance = 1.f;
  float m_currentApertureRadius = 0.0f;

  ::Vector3f m_phongLightDir = ::Vector3f(1.f, 1.f, 1.f);
  float m_phongAmbient = 0.12f;
  float m_phongDiffuse = 0.88f;
  float m_phongSpecular = 0.35f;
  float m_phongShininess = 32.f;
};