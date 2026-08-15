#include "gui.h"
#include "bitmap.h"
#include "hdrfilm.cuh"
#include "renderer.h"
#include "optix_pipeline.h"
#include "distribution.cuh"
#include "scene_loader.h"
#include "integrator_ui.h"
#include <filesystem>
#include <algorithm>
#include <functional>
#include <iostream>
#include <fstream>
#include <stb_image.h>
#include <thread>

using namespace nanogui;
using namespace futaba;
namespace fs = std::filesystem;

static constexpr float kMinFov = 5.f, kMaxFov = 120.f;
static constexpr float kMinFocusDistance = 0.1f, kMaxFocusDistance = 50.f;
static constexpr float kMinApertureRadius = 0.f, kMaxApertureRadius = 0.5f;

inline float toSlider(float v, float minV, float maxV) { return std::clamp((v - minV) / (maxV - minV), 0.f, 1.f); }
inline float fromSlider(float t, float minV, float maxV) { return minV + std::clamp(t, 0.f, 1.f) * (maxV - minV); }

static float fovToSlider(float fov) { return toSlider(fov, kMinFov, kMaxFov); }
static float sliderToFov(float t) { return fromSlider(t, kMinFov, kMaxFov); }
static float focusDistanceToSlider(float d) { return toSlider(d, kMinFocusDistance, kMaxFocusDistance); }
static float sliderToFocusDistance(float t) { return fromSlider(t, kMinFocusDistance, kMaxFocusDistance); }
static float apertureToSlider(float a) { return toSlider(a, kMinApertureRadius, kMaxApertureRadius); }
static float sliderToAperture(float t) { return fromSlider(t, kMinApertureRadius, kMaxApertureRadius); }

FutabaScreen::FutabaScreen(int width, int height, const std::string& initialScenePath)
    : Screen(nanogui::Vector2i(width, height), "Futaba Renderer") {
    cudaSetDevice(0);
    glfwGetFramebufferSize(glfwWindow(), &m_renderWidth, &m_renderHeight);

    GLFWwindow *win = glfwWindow();
    GLFWimage images[1];
    int img_w, img_h, img_channels;
    unsigned char* pixels = nullptr;
    for (const char* path : { FUTABA_SOURCE_ROOT "/assets/futaba-window.png",
                              FUTABA_SOURCE_ROOT "/assets/window.png",
                              "assets/futaba-window.png",
                              "assets/window.png" }) {
        if ((pixels = stbi_load(path, &img_w, &img_h, &img_channels, 4))) break;
    }
    
    if (pixels) {
        images[0].width = img_w;
        images[0].height = img_h;
        images[0].pixels = pixels;
        glfwSetWindowIcon(win, 1, images);
        stbi_image_free(pixels);
    }

    m_mainSettingsWindow = new Window(this, "Settings");
    m_mainSettingsWindow->setPosition(nanogui::Vector2i(15, 15));
    m_mainSettingsWindow->setLayout(new GroupLayout(10, 5, 5, 5));
    m_mainSettingsWindow->setVisible(false);
    Window *window = m_mainSettingsWindow;

    Widget *btnPanel = new Widget(window);
    btnPanel->setLayout(new BoxLayout(Orientation::Horizontal, Alignment::Fill, 3, 3));

    Button *btnReset = new Button(btnPanel, "Reset");
    btnReset->setCallback([this] { m_film->clear(); });

    Button *btnSave = new Button(btnPanel, "Save EXR");
    btnSave->setCallback([this] {
        std::string path = file_dialog({{"exr", "OpenEXR"}}, true);
        if (!path.empty()) {
            Bitmap *bmp = nullptr;
            if (m_useDenoiser && m_denoiser.getOutputBeauty() != nullptr) {
                bmp = m_film->toBitmap(m_denoiser.getOutputBeauty());
            } else {
                bmp = m_film->toBitmap();
            }
            bmp->saveEXR(path);
            delete bmp;
        }
    });

    new Label(window, "Scene", "sans-bold");
    m_sceneLabel = new Label(window, "Cornell Box (built-in)");
    m_triCountLabel = new Label(window, "Triangles: 0");

    Button *btnBrowse = new Button(window, "Load Scene...");
    btnBrowse->setCallback([this] {
        std::string path = file_dialog({{"xml", "Scene XML"}}, false);
        if (!path.empty())
            loadScene(path);
    });

    new Label(window, "Integrator", "sans-bold");
    std::vector<std::string> integratorNames;
    for (const auto& integrator : IntegratorRegistry::getIntegrators()) {
        integratorNames.push_back(integrator->getName());
    }
    m_integratorCombo = new ComboBox(window, integratorNames);
    
    int integratorIndex = 0;
    auto integrators = IntegratorRegistry::getIntegrators();
    for (size_t i = 0; i < integrators.size(); ++i) {
        if (integrators[i]->getMode() == m_integratorMode) {
            integratorIndex = (int)i;
            break;
        }
    }
    m_integratorCombo->setSelectedIndex(integratorIndex);
    m_integratorCombo->setCallback([this](int index) {
        auto activeInt = IntegratorRegistry::getIntegrators()[index];
        m_integratorMode = activeInt->getMode();
        updateIntegratorUI();
        m_film->clear();
    });

    Button *btnSettings = new Button(window, "Render Settings...");
    btnSettings->setCallback([this] {
        if (m_settingsWindow) {
            m_settingsWindow->setVisible(!m_settingsWindow->visible());
            performLayout();
        }
    });

    m_settingsWindow = new Window(this, "Rendering Settings");
    m_settingsWindow->setPosition(nanogui::Vector2i(245, 15));
    m_settingsWindow->setLayout(new GroupLayout(10, 5, 5, 5));
    m_settingsWindow->setVisible(false);

    auto *settingsCloseBtn = new Button(m_settingsWindow->buttonPanel(), "", ENTYPO_ICON_CROSS);
    settingsCloseBtn->setCallback([this] {
        m_settingsWindow->setVisible(false);
    });

    Widget *settingsGrid = new Widget(m_settingsWindow);
    settingsGrid->setLayout(new GridLayout(Orientation::Horizontal, 2, Alignment::Middle, 0, 8));

    new Label(settingsGrid, "Tonemapping", "sans-bold");
    ComboBox *tonemapCombo = new ComboBox(settingsGrid, {"None", "ACES", "Reinhard", "Filmic"});
    tonemapCombo->setSelectedIndex((int)m_tonemappingMode);
    tonemapCombo->setFixedWidth(130);
    tonemapCombo->setCallback([this](int index) {
        m_tonemappingMode = index;
    });

    new Label(settingsGrid, "Denoiser", "sans-bold");
    ComboBox *denoiserCombo = new ComboBox(settingsGrid, {"None", "OptiX Denoiser"});
    denoiserCombo->setSelectedIndex(m_useDenoiser ? 1 : 0);
    denoiserCombo->setFixedWidth(130);
    denoiserCombo->setCallback([this](int index) {
        m_useDenoiser = (index == 1);
    });

    new Label(settingsGrid, "Light Sampler", "sans-bold");
    m_lightSamplerCombo = new ComboBox(settingsGrid, {"Power"});
    m_lightSamplerCombo->setSelectedIndex((int)m_lightSamplerType);
    m_lightSamplerCombo->setFixedWidth(130);
    m_lightSamplerCombo->setCallback([this](int index) {
        m_lightSamplerType = index;
        m_film->clear();
    });

    new Label(settingsGrid, "Path Guiding", "sans-bold");

    m_guiding.renderUI(settingsGrid, m_settingsWindow,
        [this] { m_film->clear(); },
        [this] { m_trainRequested = true; });

    m_integratorSettingsWindow = new Window(this, "Integrator Settings");
    m_integratorSettingsWindow->setPosition(nanogui::Vector2i(245, 15));
    m_integratorSettingsWindow->setLayout(new GroupLayout(10, 5, 5, 5));
    m_integratorSettingsWindow->setVisible(false);

    auto *integratorCloseBtn = new Button(m_integratorSettingsWindow->buttonPanel(), "", ENTYPO_ICON_CROSS);
    integratorCloseBtn->setCallback([this] {
        m_integratorSettingsWindow->setVisible(false);
    });

    CheckBox *cbNormals = new CheckBox(window, "Use Vertex Normals");
    cbNormals->setChecked(m_useVertexNormals);
    cbNormals->setCallback([this](bool checked) {
        m_useVertexNormals = checked;
        m_scene.use_vertex_normals = checked;
        m_film->clear();
    });

    CheckBox *cbAa = new CheckBox(window, "Antialiasing");
    cbAa->setChecked(m_useAntialiasing);
    cbAa->setCallback([this](bool checked) {
        m_useAntialiasing = checked;
        m_film->clear();
    });

    CheckBox *cbNee = new CheckBox(window, "Next Event Estimation");
    cbNee->setChecked(m_useNEE);
    cbNee->setCallback([this](bool checked) {
        m_useNEE = checked;
        m_scene.use_nee = checked;
        m_film->clear();
    });

    auto createSliderControl = [&](Widget* parent, const std::string& title, float val, auto toS, auto fromS, auto setVal) -> Slider* {
        new Label(parent, title, "sans-bold");
        Widget *panel = new Widget(parent);
        panel->setLayout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 10));
        Slider *slider = new Slider(panel);
        slider->setValue(toS(val));
        slider->setFixedWidth(100);
        Label *valLabel = new Label(panel, std::to_string(val));
        slider->setCallback([this, fromS, setVal, valLabel](float t) {
            float v = fromS(t);
            setVal(v);
            valLabel->setCaption(std::to_string(v));
            updateCamera();
        });
        return slider;
    };

    new Label(window, "FOV", "sans-bold");
    m_fovSlider = new Slider(window);
    m_fovSlider->setValue(fovToSlider(m_cameraController.fov()));
    m_fovSlider->setCallback([this](float value) {
        m_cameraController.setFov(sliderToFov(value));
        updateCamera();
    });

    m_focusSlider = createSliderControl(window, "Focus Distance", m_cameraController.focusDistance(),
        focusDistanceToSlider, sliderToFocusDistance, [this](float v) { m_cameraController.setFocusDistance(v); });

    m_apertureSlider = createSliderControl(window, "Aperture", m_cameraController.apertureRadius(),
        apertureToSlider, sliderToAperture, [this](float v) { m_cameraController.setApertureRadius(v); });

    m_fpsLabel = new Label(window, "FPS: 0.0");
    m_fpsLabel->setFont("sans-bold");

    Widget *depthGridPanel = new Widget(window);
    depthGridPanel->setLayout(new GridLayout(Orientation::Horizontal, 2, Alignment::Middle, 0, 6));

    new Label(depthGridPanel, "Depth", "sans-bold");
    IntBox<int> *depthBox = new IntBox<int>(depthGridPanel, m_maxDepth);
    depthBox->setEditable(true);
    depthBox->setSpinnable(true);
    depthBox->setAlignment(TextBox::Alignment::Right);
    depthBox->setMinMaxValues(1, 64);
    depthBox->setFixedWidth(70);
    depthBox->setCallback([this](int value) {
        m_maxDepth = value;
        recreateRenderTargets(m_renderWidth, m_renderHeight);
        m_film->clear();
    });

    new Label(depthGridPanel, "RR", "sans-bold");
    IntBox<int> *rrBox = new IntBox<int>(depthGridPanel, m_rrDepth);
    rrBox->setEditable(true);
    rrBox->setSpinnable(true);
    rrBox->setAlignment(TextBox::Alignment::Right);
    rrBox->setMinMaxValues(1, 64);
    rrBox->setFixedWidth(70);
    rrBox->setCallback([this](int value) {
        m_rrDepth = value;
        m_film->clear();
    });

    setVisible(true);
    performLayout();

    glGenFramebuffers(1, &m_fbo);
    recreateRenderTargets(m_renderWidth, m_renderHeight);

    if (!initialScenePath.empty()) {
        loadScene(initialScenePath);
    } else {
        buildCornellBox(m_scene);
        m_scene.use_vertex_normals = m_useVertexNormals;
        m_scene.use_nee = m_useNEE;
        m_guiding.setSceneBounds(AABB(m_scene.boundsMin, m_scene.boundsMax));
    }

    if (m_triCountLabel)
        m_triCountLabel->setCaption("Triangles: " + std::to_string(m_scene.triangleCount));

    updateCamera();

    futaba::initOptix();
    m_denoiser.init(futaba::getOptixContext(), m_renderWidth, m_renderHeight);
    updateIntegratorUI();

    m_loadingWindow = new Window(this, "");
    m_loadingWindow->setLayout(new BoxLayout(Orientation::Vertical, Alignment::Middle, 20, 10));
    m_loadingWindow->setFixedWidth(500);

    auto* loadingTitle = new Label(m_loadingWindow, "F U T A B A", "sans-bold");
    loadingTitle->setFontSize(26);

    auto* loadingSubtitle = new Label(m_loadingWindow, "Physically-Based GPU Path Tracer", "sans");
    loadingSubtitle->setFontSize(13);

    // Scene & hardware metadata row
    Widget* infoRow = new Widget(m_loadingWindow);
    infoRow->setLayout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 8));

    std::string scName = initialScenePath.empty() ? "Cornell Box" : fs::path(initialScenePath).stem().string();
    m_loadingSceneLabel = new Label(infoRow, "Scene: " + scName + " (" + std::to_string(m_scene.triangleCount) + " tris)", "sans");
    m_loadingSceneLabel->setFontSize(12);

    auto* sepLabel = new Label(infoRow, "|", "sans-bold");
    sepLabel->setFontSize(12);

    auto* deviceLabel = new Label(infoRow, "NVIDIA OptiX 7 (RT Cores)", "sans");
    deviceLabel->setFontSize(12);

    // Progress bar + percentage counter
    Widget* progressRow = new Widget(m_loadingWindow);
    progressRow->setLayout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 10));

    m_loadingProgressBar = new ProgressBar(progressRow);
    m_loadingProgressBar->setFixedWidth(410);
    m_loadingProgressBar->setValue(0.0f);

    m_loadingPercentLabel = new Label(progressRow, "0%", "sans-bold");
    m_loadingPercentLabel->setFontSize(13);
    m_loadingPercentLabel->setFixedWidth(45);

    // Status description
    m_loadingStatusLabel = new Label(m_loadingWindow, "Initializing OptiX & CUDA context...", "sans");
    m_loadingStatusLabel->setFontSize(13);
    m_loadingStatusLabel->setFixedWidth(460);

    performLayout();
    m_loadingWindow->center();

    std::thread compileThread([]() {
        futaba::launch_initial_pipeline_compile();
    });
    compileThread.detach();
}

FutabaScreen::~FutabaScreen() {
    m_textureManager.clear();
    delete m_film;
    m_film = nullptr;
    m_scene.clear();
    if (m_cudaPboResource != nullptr) {
        cudaGraphicsUnregisterResource(m_cudaPboResource);
        m_cudaPboResource = nullptr;
    }
    if (m_fbo != 0) {
        glDeleteFramebuffers(1, &m_fbo);
        m_fbo = 0;
    }
    if (m_glPbo != 0) {
        glDeleteBuffers(1, &m_glPbo);
        m_glPbo = 0;
    }
    if (m_glTex != 0) {
        glDeleteTextures(1, &m_glTex);
        m_glTex = 0;
    }
}

bool FutabaScreen::loadScene(const std::string &xmlPath) {
    try {
        m_textureManager.clear();
        SceneLoader loader;
        CPUScene loaded;
        std::string error;

        if (!loader.load(xmlPath, loaded, error)) {
            auto dlg = new MessageDialog(this, MessageDialog::Type::Warning, "Scene load failed", error);
            (void)dlg;
            return false;
        }

        const fs::path sceneDir = fs::path(xmlPath).parent_path();
        auto bindTextures = [&](const std::vector<std::string>& paths, bool sRGB, auto setObj) {
            const size_t count = std::min(loaded.materials.size(), paths.size());
            for (size_t i = 0; i < count; ++i) {
                if (paths[i].empty()) continue;
                fs::path p = fs::path(paths[i]);
                if (p.is_relative()) p = sceneDir / p;
                cudaTextureObject_t tex = m_textureManager.createTexture(p.string(), sRGB);
                if (tex != 0) setObj(loaded.materials[i], tex);
            }
        };

        bindTextures(loaded.materialTexturePaths, true, [](Material& m, cudaTextureObject_t t) { m.texObj = t; });
        bindTextures(loaded.materialNormalMapPaths, false, [](Material& m, cudaTextureObject_t t) { m.normalMapTexObj = t; });

        m_scene.clear();
        m_scene.load(loaded);
        m_guiding.setSceneBounds(AABB(m_scene.boundsMin, m_scene.boundsMax));

        if (loaded.integratorType == "volpath" || !loaded.media.empty()) {
            m_integratorMode = futaba::INTEGRATOR_VOLPATH;
        } else {
            m_integratorMode = futaba::INTEGRATOR_PATH;
        }

        if (m_integratorCombo != nullptr) {
            auto integrators = IntegratorRegistry::getIntegrators();
            for (size_t i = 0; i < integrators.size(); ++i) {
                if (integrators[i]->getMode() == m_integratorMode) {
                    m_integratorCombo->setSelectedIndex((int)i);
                    break;
                }
            }
        }

        if (loaded.camera.hasCamera) {
            float focusDistance = length(loaded.camera.target - loaded.camera.origin);
            m_cameraController.setFromConfig(loaded.camera.origin, loaded.camera.target, loaded.camera.up,
                                            loaded.camera.fov, focusDistance,
                                            m_cameraController.apertureRadius());

            if (m_fovSlider)
                m_fovSlider->setValue(fovToSlider(m_cameraController.fov()));
            if (m_focusSlider)
                m_focusSlider->setValue(focusDistanceToSlider(m_cameraController.focusDistance()));
            if (m_apertureSlider)
                m_apertureSlider->setValue(apertureToSlider(m_cameraController.apertureRadius()));
            updateCamera();
        }

        m_sceneLabel->setCaption(fs::path(xmlPath).filename().string());
        if (m_triCountLabel)
            m_triCountLabel->setCaption("Triangles: " + std::to_string(m_scene.triangleCount));
        performLayout();
        m_film->clear();
        return true;
    } catch (const std::exception &e) {
        auto dlg = new MessageDialog(this, MessageDialog::Type::Warning, "Scene load failed", e.what());
        (void)dlg;
        return false;
    }
}

futaba::LaunchParams FutabaScreen::populateLaunchParams(uchar4* pboPtr) const {
    LaunchParams params = {};
    params.pbo_ptr = pboPtr;
    params.width = m_renderWidth;
    params.height = m_renderHeight;
    params.camera = m_camera;
    params.scene = m_scene;
    params.max_depth = m_maxDepth;
    params.rr_depth = m_rrDepth;
    params.integrator_mode = m_integratorMode;
    params.tonemapping_mode = m_tonemappingMode;
    params.use_antialiasing = m_useAntialiasing;
    params.denoise.active = m_useDenoiser;
    params.light_sampler.type = m_lightSamplerType;
    params.light_sampler.cdf.emitterTriangleCdf = m_scene.emitterTriangleCdf;
    params.light_sampler.cdf.emissiveTriangleIndices = m_scene.emissiveTriangleIndices;
    params.light_sampler.cdf.emissiveGlobalToIndex = m_scene.emissiveGlobalToIndex;
    params.light_sampler.cdf.emissiveTriCount = m_scene.emissiveTriCount;
    params.light_sampler.cdf.emitterTriangleFuncSum = m_scene.emitterTriangleFuncSum;
    m_guiding.updateLaunchParams(params);
    return params;
}

void FutabaScreen::renderLoop() {
    GLFWwindow *win = glfwWindow();
    double lastTime = glfwGetTime(), lastFrameTime = lastTime;
    int nbFrames = 0;
    bool was_compile_completed = false;

    while (!glfwWindowShouldClose(win)) {
        try {
            glfwPollEvents();

            if (!futaba::g_optixCompileCompleted.load()) {
                int fw, fh;
                glfwGetFramebufferSize(win, &fw, &fh);
                if (fw != m_renderWidth || fh != m_renderHeight) {
                    m_renderWidth = fw;
                    m_renderHeight = fh;
                    recreateRenderTargets(fw, fh);
                    performLayout();
                }

                if (m_loadingWindow) {
                    m_loadingWindow->center();

                    float target = futaba::g_optixCompileProgress.load();
                    // Smoothly lerp towards target progress
                    m_smoothProgress += (target - m_smoothProgress) * 0.12f;
                    if (target < 1.0f && m_smoothProgress < target) {
                        m_smoothProgress += 0.001f;
                    }
                    float displayVal = std::min(1.0f, std::max(0.0f, m_smoothProgress));
                    m_loadingProgressBar->setValue(displayVal);

                    int pct = (int)std::round(displayVal * 100.0f);
                    if (m_loadingPercentLabel) {
                        m_loadingPercentLabel->setCaption(std::to_string(std::min(100, pct)) + "%");
                    }
                    if (m_loadingStatusLabel) {
                        m_loadingStatusLabel->setCaption(futaba::g_optixCompileStatus.load());
                    }
                }

                glViewport(0, 0, m_renderWidth, m_renderHeight);
                glClearColor(0.09f, 0.09f, 0.11f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                drawContents();
                drawWidgets();
                glfwSwapBuffers(win);
                continue;
            } else if (!was_compile_completed) {
                was_compile_completed = true;
                if (m_loadingWindow) {
                    m_loadingWindow->setVisible(false);
                }
                if (m_mainSettingsWindow) {
                    m_mainSettingsWindow->setVisible(true);
                    m_mainSettingsWindow->requestFocus();
                }
                updateFocus(nullptr);
                performLayout();
            }

            double currentTime = glfwGetTime();
            float deltaTime = (float)(currentTime - lastFrameTime);
            lastFrameTime = currentTime;

            nbFrames++;
            if (currentTime - lastTime >= 1.0) {
                char buf[32];
                snprintf(buf, sizeof(buf), "FPS: %.1f", (double)nbFrames / (currentTime - lastTime));
                if (m_fpsLabel)
                    m_fpsLabel->setCaption(buf);
                nbFrames = 0;
                lastTime = currentTime;
            }

            bool moved = m_cameraController.update(deltaTime);

            int fw, fh;
            glfwGetFramebufferSize(win, &fw, &fh);
            if (fw != m_renderWidth || fh != m_renderHeight) {
                m_renderWidth = fw;
                m_renderHeight = fh;
                recreateRenderTargets(fw, fh);
                moved = true;
            }

            if (moved)
                updateCamera();

            uchar4 *d_pbo_ptr = nullptr;
            size_t num_bytes = 0;
            cudaGraphicsMapResources(1, &m_cudaPboResource, g_pipeline.renderStream);
            cudaGraphicsResourceGetMappedPointer((void **)&d_pbo_ptr, &num_bytes, m_cudaPboResource);

            LaunchParams params = populateLaunchParams(d_pbo_ptr);

            // Training pass: capture one frame of path samples into the GPU
            const bool autoTraining = m_guiding.isTrainingEnabled();
            const bool capturing = (m_trainRequested || autoTraining) && m_guiding.enabled();
            if (capturing) {
                m_trainingBuffers.allocate(m_renderWidth, m_renderHeight, m_maxDepth);
                m_trainingBuffers.clear();
                params.guiding.train_active    = m_trainingBuffers.active();
                params.guiding.train_position  = m_trainingBuffers.position();
                params.guiding.train_wo        = m_trainingBuffers.wo();
                params.guiding.train_radiance  = m_trainingBuffers.radiance();
                params.guiding.train_pdf       = m_trainingBuffers.pdf();
                params.guiding.train_max_depth = m_maxDepth;
            }

            auto activeInt = IntegratorRegistry::getIntegrator(m_integratorMode);
            if (activeInt) {
                activeInt->updateLaunchParams(params);
            }

            launch_render(m_film, &m_denoiser, params);

            if (capturing) {
                m_guiding.train(m_trainingBuffers, m_maxDepth);
                if (m_trainRequested) {
                    m_film->clear();
                    m_trainRequested = false;
                }
            }

            cudaGraphicsUnmapResources(1, &m_cudaPboResource, g_pipeline.renderStream);

            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_glPbo);
            glBindTexture(GL_TEXTURE_2D, m_glTex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_renderWidth, m_renderHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            glBindFramebuffer(GL_READ_FRAMEBUFFER, m_fbo);
            glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_glTex, 0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
            glfwGetFramebufferSize(win, &fw, &fh);
            glBlitFramebuffer(0, 0, m_renderWidth, m_renderHeight, 0, fh, fw, 0, GL_COLOR_BUFFER_BIT, GL_NEAREST);

            drawContents();
            drawGizmo();
            drawWidgets();
            glfwSwapBuffers(win);
        } catch (const std::exception& e) {
            auto dlg = new MessageDialog(this, MessageDialog::Type::Warning, "Error", e.what());
            (void)dlg;
        }
    }
}

bool FutabaScreen::keyboardEvent(int key, int scancode, int action, int modifiers) {
    if (Screen::keyboardEvent(key, scancode, action, modifiers))
        return true;
    m_cameraController.handleKeyboard(key, action);
    return true;
}

bool FutabaScreen::mouseButtonEvent(const nanogui::Vector2i &p, int button, bool down, int modifiers) {
    if (Screen::mouseButtonEvent(p, button, down, modifiers))
        return true;
    return m_cameraController.handleMouseButton(button, down);
}

bool FutabaScreen::mouseMotionEvent(const nanogui::Vector2i &p, const nanogui::Vector2i &rel, int button, int modifiers) {
    if (Screen::mouseMotionEvent(p, rel, button, modifiers))
        return true;
    if (m_cameraController.handleMouseMotion(rel)) {
        updateCamera();
        return true;
    }
    return false;
}

bool FutabaScreen::scrollEvent(const nanogui::Vector2i &p, const nanogui::Vector2f &rel) {
    if (Screen::scrollEvent(p, rel))
        return true;
    m_cameraController.handleScroll(rel);
    if (m_fovSlider)
        m_fovSlider->setValue(fovToSlider(m_cameraController.fov()));

    updateCamera();
    return true;
}

bool FutabaScreen::resizeEvent(const nanogui::Vector2i &size) {
    if (!Screen::resizeEvent(size))
        return false;

    int fw, fh;
    glfwGetFramebufferSize(glfwWindow(), &fw, &fh);
    if (fw <= 0 || fh <= 0)
        return true;

    m_renderWidth = fw;
    m_renderHeight = fh;
    recreateRenderTargets(fw, fh);
    updateCamera();

    return true;
}

void FutabaScreen::recreateRenderTargets(int width, int height) {
    if (m_cudaPboResource != nullptr) {
        cudaGraphicsUnregisterResource(m_cudaPboResource);
        m_cudaPboResource = nullptr;
    }

    if (m_glPbo != 0) {
        glDeleteBuffers(1, &m_glPbo);
        m_glPbo = 0;
    }

    if (m_glTex != 0) {
        glDeleteTextures(1, &m_glTex);
        m_glTex = 0;
    }

    glGenTextures(1, &m_glTex);
    glBindTexture(GL_TEXTURE_2D, m_glTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenBuffers(1, &m_glPbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_glPbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4 * sizeof(GLubyte), NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&m_cudaPboResource, m_glPbo, cudaGraphicsMapFlagsWriteDiscard);

    if (m_film)
        delete m_film;
    m_film = new HDRFilm(width, height);

    if (futaba::getOptixContext()) {
        m_denoiser.resize(width, height);
    }
}

void FutabaScreen::updateCamera() {
    m_cameraController.updateCamera(m_camera, m_renderWidth, m_renderHeight);
    m_film->clear();
}

void FutabaScreen::drawGizmo() {
    NVGcontext *vg = mNVGContext;

    int size = 80;
    int margin = 20;
    int centerX = mSize.x() - size / 2 - margin;
    int centerY = size / 2 + margin;

    nvgSave(vg);

    nvgBeginPath(vg);
    nvgCircle(vg, (float)centerX, (float)centerY, size / 2.0f);
    nvgFillColor(vg, nvgRGBA(40, 40, 40, 150));
    nvgFill(vg);

    auto drawAxis = [&](const futaba::Vector3f &worldDir, NVGcolor color, const char *label) {
        float dx = dot(worldDir, m_camera.right);
        float dy = -dot(worldDir, m_camera.trueUp);

        float len = size * 0.35f;
        float endX = centerX + dx * len;
        float endY = centerY + dy * len;

        nvgBeginPath(vg);
        nvgMoveTo(vg, (float)centerX, (float)centerY);
        nvgLineTo(vg, endX, endY);
        nvgStrokeColor(vg, color);
        nvgStrokeWidth(vg, 2.0f);
        nvgStroke(vg);

        nvgBeginPath(vg);
        nvgCircle(vg, endX, endY, 8.0f);
        nvgFillColor(vg, color);
        nvgFill(vg);

        nvgFontSize(vg, 14.0f);
        nvgFontFace(vg, "sans-bold");
        nvgTextAlign(vg, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);
        nvgFillColor(vg, nvgRGBA(255, 255, 255, 255));
        nvgText(vg, endX, endY, label, nullptr);
    };

    drawAxis(futaba::Vector3f(1, 0, 0), nvgRGBA(255, 50, 50, 255), "X");
    drawAxis(futaba::Vector3f(0, 1, 0), nvgRGBA(50, 255, 50, 255), "Y");
    drawAxis(futaba::Vector3f(0, 0, 1), nvgRGBA(50, 50, 255, 255), "Z");

    nvgRestore(vg);
}

void FutabaScreen::updateIntegratorUI() {
    if (!m_integratorSettingsWindow) return;

    m_integratorSettingsWindow->setVisible(false);

    std::vector<Widget *> childrenToRemove;
    for (auto *child : m_integratorSettingsWindow->children()) {
        if (child != m_integratorSettingsWindow->buttonPanel()) {
            childrenToRemove.push_back(child);
        }
    }
    for (auto *child : childrenToRemove) {
        m_integratorSettingsWindow->removeChild(child);
    }

    auto activeInt = IntegratorRegistry::getIntegrator(m_integratorMode);
    if (activeInt) {
        activeInt->renderUI(m_integratorSettingsWindow, [this] { m_film->clear(); });

        int nonPanelChildCount = 0;
        for (auto *child : m_integratorSettingsWindow->children()) {
            if (child != m_integratorSettingsWindow->buttonPanel()) {
                nonPanelChildCount++;
            }
        }

        if (nonPanelChildCount > 0) {
            m_integratorSettingsWindow->setTitle(activeInt->getName() + " Controls");
            m_integratorSettingsWindow->setVisible(true);
        }
    }
    performLayout();
}

void FutabaScreen::handleException(const std::exception &e, const std::string &title) {
    auto dlg = new MessageDialog(this, MessageDialog::Type::Warning, title, e.what());
    (void)dlg;
}
