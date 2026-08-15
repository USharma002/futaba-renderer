#pragma once

#include <nanogui/nanogui.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "perspective.cuh"
#include "renderer.h"
#include "scene.cuh"
#include "scene_loader.h"
#include "texture_manager.h"
#include "denoiser.h"
#include "camera_controller.h"
#include "guiding.h"

FUTABA_NAMESPACE_BEGIN
class HDRFilm;
FUTABA_NAMESPACE_END

class FutabaScreen : public nanogui::Screen {
public:
    FutabaScreen(int width, int height, const std::string& initialScenePath = "");
    virtual ~FutabaScreen() override;
    
    void renderLoop();

protected:
    virtual bool keyboardEvent(int key, int scancode, int action, int modifiers) override;
    virtual bool mouseButtonEvent(const nanogui::Vector2i &p, int button, bool down, int modifiers) override;
    virtual bool mouseMotionEvent(const nanogui::Vector2i &p, const nanogui::Vector2i &rel, int button, int modifiers) override;
    virtual bool scrollEvent(const nanogui::Vector2i &p, const nanogui::Vector2f &rel) override;
    virtual bool resizeEvent(const nanogui::Vector2i &size) override;

    void updateCamera();
    bool loadScene(const std::string &xmlPath);
    void recreateRenderTargets(int width, int height);
    void drawGizmo();
    futaba::LaunchParams populateLaunchParams(uchar4* pboPtr) const;

private:
    int m_renderWidth;
    int m_renderHeight;

    nanogui::Label *m_fpsLabel = nullptr;
    nanogui::Label *m_sceneLabel = nullptr;
    nanogui::Label *m_triCountLabel = nullptr;
    nanogui::Window *m_integratorSettingsWindow = nullptr;
    nanogui::Window *m_settingsWindow = nullptr;
    nanogui::Window *m_mainSettingsWindow = nullptr;
    nanogui::Slider *m_fovSlider = nullptr;
    nanogui::Slider *m_focusSlider = nullptr;
    nanogui::Slider *m_apertureSlider = nullptr;
    nanogui::ComboBox *m_integratorCombo = nullptr;
    nanogui::ComboBox *m_lightSamplerCombo = nullptr;
    
    int m_maxDepth = 12;
    int m_rrDepth = 5;
    int m_integratorMode = futaba::INTEGRATOR_PATH;
    int m_tonemappingMode = futaba::TONEMAPPING_NONE;
    int m_lightSamplerType = futaba::LIGHT_SAMPLER_POWER;
    bool m_useVertexNormals = true;
    bool m_useAntialiasing = true;
    bool m_useNEE = true;
    bool m_useDenoiser = false;
    bool m_trainRequested = false;

    futaba::DenoiserManager m_denoiser;
    futaba::GuidingManager m_guiding;
    futaba::TrainingBufferManager m_trainingBuffers;

    GLuint m_glTex = 0;
    GLuint m_glPbo = 0;
    GLuint m_fbo = 0;
    cudaGraphicsResource_t m_cudaPboResource = nullptr;

    futaba::HDRFilm *m_film = nullptr;
    futaba::PerspectiveCamera m_camera;
    futaba::Scene m_scene;

    // Camera state + WASD/mouse-look navigation, shared with no duplicate copy.
    futaba::CameraController m_cameraController;

    void updateIntegratorUI();
    void handleException(const std::exception &e, const std::string &title = "Error");

    futaba::TextureManager m_textureManager;

    // Loading splash overlay for background OptiX pipeline compilation
    nanogui::Window*      m_loadingWindow = nullptr;
    nanogui::ProgressBar* m_loadingProgressBar = nullptr;
    nanogui::Label*       m_loadingStatusLabel = nullptr;
    nanogui::Label*       m_loadingPercentLabel = nullptr;
    nanogui::Label*       m_loadingSceneLabel = nullptr;
    float                 m_smoothProgress = 0.0f;
};