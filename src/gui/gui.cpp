#include "gui.h"
#include "bitmap.h"
#include "hdrfilm.cuh"
#include "renderer.h"
#include "distribution.cuh"
#include "scene_loader.h"
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

static cudaTextureObject_t createCudaTexture(const std::string& filename,
                                             std::vector<cudaArray*>& allocatedArrays,
                                             std::vector<cudaTextureObject_t>& allocatedTextures);

static const std::vector<std::string> g_allBufferNames = {
    "Active",
    "Position",
    "Normals",
    "Incoming Angle (wi)",
    "Outgoing Angle (wo)",
    "Incoming Radiance",
    "Material ID"
};

static constexpr float kMinFov = 5.f;
static constexpr float kMaxFov = 120.f;
static constexpr float kMinFocusDistance = 0.1f;
static constexpr float kMaxFocusDistance = 50.f;
static constexpr float kMinApertureRadius = 0.f;
static constexpr float kMaxApertureRadius = 0.5f;
static constexpr float kMinPhongStrength = 0.f;
static constexpr float kMaxPhongStrength = 2.f;
static constexpr float kMinPhongShininess = 1.f;
static constexpr float kMaxPhongShininess = 128.f;

static float fovToSlider(float fov) {
    float t = (fov - kMinFov) / (kMaxFov - kMinFov);
    if (t < 0.f)
        t = 0.f;
    if (t > 1.f)
        t = 1.f;
    return t;
}

static float sliderToFov(float t) {
    if (t < 0.f)
        t = 0.f;
    if (t > 1.f)
        t = 1.f;
    return kMinFov + t * (kMaxFov - kMinFov);
}

static float focusDistanceToSlider(float focusDistance) {
    float t = (focusDistance - kMinFocusDistance) /
              (kMaxFocusDistance - kMinFocusDistance);
    if (t < 0.f)
        t = 0.f;
    if (t > 1.f)
        t = 1.f;
    return t;
}

static float sliderToFocusDistance(float t) {
    if (t < 0.f)
        t = 0.f;
    if (t > 1.f)
        t = 1.f;
    return kMinFocusDistance + t * (kMaxFocusDistance - kMinFocusDistance);
}

static float apertureToSlider(float apertureRadius) {
    float t = (apertureRadius - kMinApertureRadius) /
              (kMaxApertureRadius - kMinApertureRadius);
    if (t < 0.f)
        t = 0.f;
    if (t > 1.f)
        t = 1.f;
    return t;
}

static float sliderToAperture(float t) {
    if (t < 0.f)
        t = 0.f;
    if (t > 1.f)
        t = 1.f;
    return kMinApertureRadius + t * (kMaxApertureRadius - kMinApertureRadius);
}

static float toUnitRange(float value, float minVal, float maxVal) {
    float t = (value - minVal) / (maxVal - minVal);
    if (t < 0.f)
        t = 0.f;
    if (t > 1.f)
        t = 1.f;
    return t;
}

static float fromUnitRange(float t, float minVal, float maxVal) {
    if (t < 0.f)
        t = 0.f;
    if (t > 1.f)
        t = 1.f;
    return minVal + t * (maxVal - minVal);
}

FutabaScreen::FutabaScreen(int width, int height)
        : Screen(nanogui::Vector2i(width, height), "Futaba Renderer") {
    // Initialize render size from framebuffer
    glfwGetFramebufferSize(glfwWindow(), &m_renderWidth, &m_renderHeight);

    // Set window icon
    GLFWwindow *win = glfwWindow();
    GLFWimage images[1];
    int img_w, img_h, img_channels;
    
    // Try absolute paths using FUTABA_SOURCE_ROOT first, falling back to relative paths
    unsigned char* pixels = stbi_load(FUTABA_SOURCE_ROOT "/assets/futaba-window.png", &img_w, &img_h, &img_channels, 4);
    if (!pixels) {
        pixels = stbi_load(FUTABA_SOURCE_ROOT "/assets/window.png", &img_w, &img_h, &img_channels, 4);
    }
    if (!pixels) {
        pixels = stbi_load("assets/futaba-window.png", &img_w, &img_h, &img_channels, 4);
    }
    if (!pixels) {
        pixels = stbi_load("assets/window.png", &img_w, &img_h, &img_channels, 4);
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
    btnPanel->setLayout(
            new BoxLayout(Orientation::Horizontal, Alignment::Fill, 3, 3));

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
    m_integratorCombo =
            new ComboBox(window, {"Path", "Normals", "Depth", "Albedo", "Phong", "Primitives", "Heatmap", "VolPath"});
    m_integratorCombo->setSelectedIndex((int)m_integratorMode);
    m_integratorCombo->setCallback([this](int index) {
        m_integratorMode = index;
        if (m_phongWindow)
            m_phongWindow->setVisible(index == futaba::INTEGRATOR_PHONG);
        performLayout();
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

        // Close button
        auto *settingsCloseBtn = new Button(m_settingsWindow->buttonPanel(), "", ENTYPO_ICON_CROSS);
        settingsCloseBtn->setCallback([this] {
            m_settingsWindow->setVisible(false);
        });

        Widget *settingsGrid = new Widget(m_settingsWindow);
        settingsGrid->setLayout(new GridLayout(Orientation::Horizontal, 2, Alignment::Middle, 0, 8));

        // 1. Tonemapping dropdown
        new Label(settingsGrid, "Tonemapping", "sans-bold");
        ComboBox *tonemmapCombo = new ComboBox(settingsGrid, {"None", "ACES", "Reinhard", "Filmic"});
        tonemmapCombo->setSelectedIndex((int)m_tonemappingMode);
        tonemmapCombo->setFixedWidth(130);
        tonemmapCombo->setCallback([this](int index) {
            m_tonemappingMode = index;
        });

        // 2. Denoiser dropdown (shifted from the main sidebar)
        new Label(settingsGrid, "Denoiser", "sans-bold");
        ComboBox *denoiserCombo = new ComboBox(settingsGrid, {"None", "OptiX Denoiser"});
        denoiserCombo->setSelectedIndex(m_useDenoiser ? 1 : 0);
        denoiserCombo->setFixedWidth(130);
        denoiserCombo->setCallback([this](int index) {
            m_useDenoiser = (index == 1);
        });

        // 3. Path Guiding dropdown (skeleton setup)
        new Label(settingsGrid, "Path Guiding", "sans-bold");
        ComboBox *guidingCombo = new ComboBox(settingsGrid, {"None", "SD-Tree (PPG)", "NPM (Neural)"});
        guidingCombo->setSelectedIndex((int)m_pathGuidingMode);
        guidingCombo->setFixedWidth(130);

        // 4. Collect Training Data checkbox
        new Label(settingsGrid, "Training Data", "sans-bold");
        CheckBox *cbTraining = new CheckBox(settingsGrid, "Collect");
        cbTraining->setChecked(m_collectTraining);
        cbTraining->setCallback([this](bool checked) {
            m_collectTraining = checked;
            m_film->clear();
        });

        // 5. Light Sampler dropdown
        new Label(settingsGrid, "Light Sampler", "sans-bold");
        m_lightSamplerCombo = new ComboBox(settingsGrid, {"Uniform", "Power"});
        m_lightSamplerCombo->setSelectedIndex((int)m_lightSamplerType);
        m_lightSamplerCombo->setFixedWidth(130);
        m_lightSamplerCombo->setCallback([this](int index) {
            m_lightSamplerType = index;
            m_film->clear();
        });

        guidingCombo->setCallback([this, cbTraining, guidingCombo](int index) {
            m_pathGuidingMode = index;
            if (index != 0 && !m_collectTraining) {
                m_collectTraining = true;
                cbTraining->setChecked(true);
            }
            updateVisualizerDropdown();
            m_film->clear();
        });

        new Widget(settingsGrid);
        Button *btnVisualizer = new Button(settingsGrid, "Buffer Visualizer...");
        btnVisualizer->setCallback([this] {
            if (m_visualizationWindow) {
                m_showVisualizer = !m_visualizationWindow->visible();
                m_visualizationWindow->setVisible(m_showVisualizer);
            }
        });

    m_phongWindow = new Window(this, "Phong Controls");
    m_phongWindow->setPosition(nanogui::Vector2i(245, 15));
    m_phongWindow->setLayout(new GroupLayout(10, 5, 5, 5));

    // Close button for phong window
    auto *phongCloseBtn = new Button(m_phongWindow->buttonPanel(), "", ENTYPO_ICON_CROSS);
    phongCloseBtn->setCallback([this] {
        m_phongWindow->setVisible(false);
    });


    auto addPhongSlider = [this](const std::string &name, float minVal,
                                 float maxVal, float initialValue,
                                 const std::function<void(float)> &onChange) {
        new Label(m_phongWindow, name, "sans-bold");
        Widget *panel = new Widget(m_phongWindow);
        panel->setLayout(
                new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 10));

        Slider *slider = new Slider(panel);
        slider->setFixedWidth(120);
        slider->setValue(toUnitRange(initialValue, minVal, maxVal));

        Label *valueLabel = new Label(panel, std::to_string(initialValue));
        slider->setCallback([this, minVal, maxVal, valueLabel, onChange](float t) {
            const float value = fromUnitRange(t, minVal, maxVal);
            valueLabel->setCaption(std::to_string(value));
            onChange(value);
            m_film->clear();
        });
    };

    addPhongSlider("Ambient", kMinPhongStrength, kMaxPhongStrength,
                   m_phongAmbient, [this](float v) { m_phongAmbient = v; });
    addPhongSlider("Diffuse", kMinPhongStrength, kMaxPhongStrength,
                   m_phongDiffuse, [this](float v) { m_phongDiffuse = v; });
    addPhongSlider("Specular", kMinPhongStrength, kMaxPhongStrength,
                   m_phongSpecular, [this](float v) { m_phongSpecular = v; });
    addPhongSlider("Shininess", kMinPhongShininess, kMaxPhongShininess,
                   m_phongShininess, [this](float v) { m_phongShininess = v; });

    m_phongWindow->setVisible(m_integratorMode == futaba::INTEGRATOR_PHONG);

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



    new Label(window, "FOV", "sans-bold");
    m_fovSlider = new Slider(window);
    m_fovSlider->setValue(fovToSlider(m_currentFov));
    m_fovSlider->setCallback([this](float value) {
        m_currentFov = sliderToFov(value);
        updateCamera();
    });

    new Label(window, "Focus Distance", "sans-bold");
    Widget *focusPanel = new Widget(window);
    focusPanel->setLayout(
            new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 10));
    m_focusSlider = new Slider(focusPanel);
    m_focusSlider->setValue(focusDistanceToSlider(m_currentFocusDistance));
    m_focusSlider->setFixedWidth(100);
    Label *focusVal = new Label(focusPanel, std::to_string(m_currentFocusDistance));
    m_focusSlider->setCallback([this, focusVal](float value) {
        m_currentFocusDistance = sliderToFocusDistance(value);
        focusVal->setCaption(std::to_string(m_currentFocusDistance));
        updateCamera();
    });

    new Label(window, "Aperture", "sans-bold");
    Widget *aperturePanel = new Widget(window);
    aperturePanel->setLayout(
            new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 10));
    m_apertureSlider = new Slider(aperturePanel);
    m_apertureSlider->setValue(apertureToSlider(m_currentApertureRadius));
    m_apertureSlider->setFixedWidth(100);
    Label *apertureVal = new Label(aperturePanel, std::to_string(m_currentApertureRadius));
    m_apertureSlider->setCallback([this, apertureVal](float value) {
        m_currentApertureRadius = sliderToAperture(value);
        apertureVal->setCaption(std::to_string(m_currentApertureRadius));
        updateCamera();
    });
    
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
        if (m_visDepthBox) {
            m_visDepthBox->setMinMaxValues(0, m_maxDepth - 1);
            if (m_visDepth >= m_maxDepth) {
                m_visDepth = m_maxDepth - 1;
                m_visDepthBox->setValue(m_visDepth);
            }
        }
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

    m_visualizationWindow = new Window(this, "Buffer Visualizer");
    m_visualizationWindow->setPosition(nanogui::Vector2i(mSize.x() - 340, 120));
    m_visualizationWindow->setLayout(new GroupLayout(10, 5, 5, 5));
    m_visualizationWindow->setVisible(false);

    auto *visCloseBtn = new Button(m_visualizationWindow->buttonPanel(), "", ENTYPO_ICON_CROSS);
    visCloseBtn->setCallback([this] {
        m_showVisualizer = false;
        if (m_visualizationWindow)
            m_visualizationWindow->setVisible(false);
    });

    new Label(m_visualizationWindow, "Select Buffer Channel", "sans-bold");
    m_visCombo = new ComboBox(m_visualizationWindow);
    m_visCombo->setCallback([this](int index) {
        std::string selected = m_visCombo->items()[index];
        auto it = std::find(g_allBufferNames.begin(), g_allBufferNames.end(), selected);
        if (it != g_allBufferNames.end()) {
            m_visBufferType = (int)std::distance(g_allBufferNames.begin(), it);
        }
    });
    updateVisualizerDropdown();

    Widget* visDepthPanel = new Widget(m_visualizationWindow);
    visDepthPanel->setLayout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 10));
    new Label(visDepthPanel, "Bounce Depth", "sans-bold");
    m_visDepthBox = new IntBox<int>(visDepthPanel, m_visDepth);
    m_visDepthBox->setEditable(true);
    m_visDepthBox->setSpinnable(true);
    m_visDepthBox->setMinMaxValues(0, m_maxDepth - 1);
    m_visDepthBox->setFixedWidth(60);
    m_visDepthBox->setCallback([this](int value) {
        m_visDepth = value;
    });

    m_visImageView = new ImageView(m_visualizationWindow, m_glTexVis);
    m_visImageView->setFixedSize(nanogui::Vector2i(300, 300));

    Button *btnSaveTrain = new Button(m_visualizationWindow, "Save Training Data...");
    btnSaveTrain->setCallback([this] {
        std::string path = file_dialog({{"bin", "Binary file"}}, true);
        if (!path.empty()) {
            saveTrainingData(path);
        }
    });

    setVisible(true);
    performLayout();

    glGenFramebuffers(1, &m_fbo);
    recreateRenderTargets(m_renderWidth, m_renderHeight);
    m_camera =
            PerspectiveCamera(m_currentFov, (float)m_renderWidth / m_renderHeight);

    m_camPos =
            ::Vector3f(m_camera.position.x, m_camera.position.y, m_camera.position.z);
    m_camForward =
            ::Vector3f(m_camera.forward.x, m_camera.forward.y, m_camera.forward.z);
    m_camUp = ::Vector3f(m_camera.trueUp.x, m_camera.trueUp.y, m_camera.trueUp.z);

    // Load Cornell box as default scene
    buildCornellBox(m_scene);
    m_scene.use_vertex_normals = m_useVertexNormals;
    m_scene.use_nee = m_useNEE;

    if (m_triCountLabel)
        m_triCountLabel->setCaption("Triangles: " +
                                                                std::to_string(m_scene.triangleCount));

    // Ensure camera is fully initialized with correct aspect ratio and lens settings
    updateCamera();

    // Initialize OptiX context early so we can initialize the denoiser manager
    futaba::initOptix();
    m_denoiser.init(futaba::getOptixContext(), m_renderWidth, m_renderHeight);
    m_guiding.init(m_renderWidth, m_renderHeight);

    // Create the centered loading window
    m_loadingWindow = new Window(this, "OptiX Initialization");
    m_loadingWindow->setLayout(new BoxLayout(Orientation::Vertical, Alignment::Middle, 20, 15));
    m_loadingWindow->setFixedWidth(450);
    
    // Title
    auto* loadingTitle = new Label(m_loadingWindow, "Futaba Path Tracer", "sans-bold");
    loadingTitle->setFontSize(24);
    
    // Status text label
    m_loadingStatusLabel = new Label(m_loadingWindow, "Initializing OptiX Pipeline...");
    m_loadingStatusLabel->setFontSize(16);
    
    // Progress Bar
    m_loadingProgressBar = new ProgressBar(m_loadingWindow);
    m_loadingProgressBar->setFixedWidth(400);
    m_loadingProgressBar->setValue(0.0f);

    m_loadingWindow->setPosition(nanogui::Vector2i((m_renderWidth - 450) / 2, (m_renderHeight - 180) / 2));
    performLayout();

    // Start background thread for OptiX pipeline compilation
    std::thread compileThread([]() {
        futaba::launch_initial_pipeline_compile();
    });
    compileThread.detach();
}

FutabaScreen::~FutabaScreen() {
    clearTextures();
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

    freeTrainingBuffers();
    if (m_cudaPboResourceVis != nullptr) {
        cudaGraphicsUnregisterResource(m_cudaPboResourceVis);
        m_cudaPboResourceVis = nullptr;
    }
    if (m_glPboVis != 0) {
        glDeleteBuffers(1, &m_glPboVis);
        m_glPboVis = 0;
    }
    if (m_glTexVis != 0) {
        glDeleteTextures(1, &m_glTexVis);
        m_glTexVis = 0;
    }
}

bool FutabaScreen::loadScene(const std::string &xmlPath) {
    try {
        clearTextures();
    SceneLoader loader;
    LoadedScene loaded;
    std::string error;

    if (!loader.load(xmlPath, loaded, error)) {
        auto dlg = new MessageDialog(this, MessageDialog::Type::Warning,
                                                                 "Scene load failed", error);
        (void)dlg;
        return false;
    }

    m_scene.clear();

    std::string baseDir = fs::path(xmlPath).parent_path().string();
    if (baseDir.empty()) baseDir = ".";
    for (size_t i = 0; i < loaded.materials.size(); ++i) {
        if (i < loaded.materialTexturePaths.size() && !loaded.materialTexturePaths[i].empty()) {
            fs::path texPath = fs::path(baseDir) / loaded.materialTexturePaths[i];
            cudaTextureObject_t texObj = createCudaTexture(texPath.string(), m_cudaTextureArrays, m_cudaTextureObjects);
            loaded.materials[i].texObj = texObj;
        }
    }

    m_scene.setTriangles(loaded.triangles.data(),
                                             (uint32_t)loaded.triangles.size());
    m_scene.setMaterials(loaded.materials.data(),
                                             (uint32_t)loaded.materials.size());
    
    std::vector<futaba::EmitterGPU> emittersGPU;
    emittersGPU.reserve(loaded.emitters.size());
    for (const auto& emitter : loaded.emitters) {
        futaba::EmitterGPU g;
        g.type = static_cast<uint32_t>(emitter.type);
        g.flags = emitter.twoSided ? futaba::EMITTER_FLAG_TWO_SIDED : 0u;
        g.radiance = emitter.radiance;
        g.position = emitter.position;
        g.direction = emitter.direction;
        g.attachedMeshId = -1;
        emittersGPU.push_back(g);
    }

    // Convert and upload mesh instances
    std::vector<futaba::MeshInstanceGPU> meshGPU;
    for (size_t i = 0; i < loaded.meshes.size(); ++i) {
        const auto& mesh = loaded.meshes[i];
        futaba::MeshInstanceGPU m;
        m.triangleStart = mesh.triangleStart;
        m.triangleCount = mesh.triangleCount;
        m.emitterId = mesh.emitterId;
        meshGPU.push_back(m);

        if (mesh.emitterId >= 0 && mesh.emitterId < (int)emittersGPU.size()) {
            emittersGPU[mesh.emitterId].attachedMeshId = (int)i;
        }
    }
    m_scene.setMeshes(meshGPU.data(), (uint32_t)meshGPU.size());
    m_scene.setEmitters(emittersGPU.data(), (uint32_t)emittersGPU.size());

    // Build emissive-triangle distribution (area * emission luminance)
    std::vector<int> emissiveTriIndices;
    std::vector<float> emissiveWeights;
    emissiveTriIndices.reserve(loaded.triangles.size());
    emissiveWeights.reserve(loaded.triangles.size());

    auto luminance = [](const Color3f &c) {
        return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
    };

    auto triangleEmission = [&](const Triangle &t) -> Color3f {
        if (t.mesh_id >= 0 && t.mesh_id < (int)loaded.meshes.size()) {
            const int emitterId = loaded.meshes[t.mesh_id].emitterId;
            if (emitterId >= 0 && emitterId < (int)loaded.emitters.size()) {
                return loaded.emitters[emitterId].radiance;
            }
        }

        if (t.material_id >= 0 && t.material_id < (int)loaded.materials.size()) {
            return loaded.materials[t.material_id].emission;
        }

        return Color3f(0.f);
    };

    for (size_t i = 0; i < loaded.triangles.size(); ++i) {
        const Triangle &t = loaded.triangles[i];
        float area = t.area();
        Color3f emission = triangleEmission(t);
        float w = area * luminance(emission);
        if (w > 0.f) {
            emissiveTriIndices.push_back((int)i);
            emissiveWeights.push_back(w);
        }
    }

    if (!emissiveWeights.empty()) {
        futaba::Distribution1D dist;
        dist.build(emissiveWeights);
        // Build global->emissive index map
        std::vector<int> globalToEmissive(loaded.triangles.size(), -1);
        for (size_t i = 0; i < emissiveTriIndices.size(); ++i) {
            int g = emissiveTriIndices[i];
            if (g >= 0 && g < (int)globalToEmissive.size())
                globalToEmissive[g] = (int)i;
        }
        // Upload cdf and index list to device (also provide global mapping)
        m_scene.setEmitterTriangleDistribution(dist.cdfData(), (int)dist.cdf.size(), dist.funcSum, emissiveTriIndices.data(), (int)emissiveTriIndices.size(), globalToEmissive.data(), (int)globalToEmissive.size());
    } else {
        m_scene.setEmitterTriangleDistribution(nullptr, 0, 0.f, nullptr, 0, nullptr, 0);
    }

    // Upload non-area (point / directional) emitter indices for NEE sampling
    {
        std::vector<int> nonAreaIndices;
        for (size_t i = 0; i < emittersGPU.size(); ++i) {
            const uint32_t t = emittersGPU[i].type;
            if (t == futaba::kEmitterTypePoint || t == futaba::kEmitterTypeDirectional)
                nonAreaIndices.push_back((int)i);
        }
        if (!nonAreaIndices.empty())
            m_scene.setNonAreaEmitters(nonAreaIndices.data(), (int)nonAreaIndices.size());
        else
            m_scene.setNonAreaEmitters(nullptr, 0);
    }

    if (loaded.hasEnvMap) {
        m_scene.setEnvironmentMap(loaded.envMapPixels.data(),
                                  (uint32_t)loaded.envMapWidth,
                                  (uint32_t)loaded.envMapHeight,
                                  loaded.envMapToWorld);
    } else if (loaded.hasConstantEnv) {
        m_scene.setConstantEnvironment(loaded.constantEnv);
    } else {
        m_scene.setEnvironmentMap(nullptr, 0, 0, ::Matrix4f());
    }
    
    m_scene.use_vertex_normals = m_useVertexNormals;
    m_scene.use_nee = m_useNEE;

    m_scene.hasMedium = loaded.hasMedium;
    m_scene.mediumMeshId = loaded.mediumMeshId;
    m_scene.medium = Medium(MEDIUM_HOMOGENEOUS, HomogeneousMedium(loaded.mediumSigmaS, loaded.mediumSigmaA, loaded.mediumG));

    if (loaded.integratorType == "volpath") {
        m_integratorMode = futaba::INTEGRATOR_VOLPATH;
    } else {
        m_integratorMode = futaba::INTEGRATOR_PATH;
    }
    if (m_integratorCombo != nullptr) {
        m_integratorCombo->setSelectedIndex(m_integratorMode);
    }

    if (loaded.hasCamera) {
        int fw, fh;
        glfwGetFramebufferSize(glfwWindow(), &fw, &fh);
        float currentAspect = (float)fw / (float)fh;
        ::Vector3f toTarget(loaded.camTarget.x - loaded.camOrigin.x,
                            loaded.camTarget.y - loaded.camOrigin.y,
                            loaded.camTarget.z - loaded.camOrigin.z);
        float loadedFocusDistance = toTarget.length();
        if (loadedFocusDistance > 0.f)
            m_currentFocusDistance = loadedFocusDistance;

        m_camera.init(loaded.camOrigin, loaded.camTarget, loaded.camUp,
                                    loaded.camFov, currentAspect,
                                    m_currentFocusDistance,
                                    m_currentApertureRadius);
        m_currentFov = loaded.camFov;
        if (m_fovSlider)
            m_fovSlider->setValue(fovToSlider(m_currentFov));
        if (m_focusSlider)
            m_focusSlider->setValue(focusDistanceToSlider(m_currentFocusDistance));
        if (m_apertureSlider)
            m_apertureSlider->setValue(apertureToSlider(m_currentApertureRadius));
        m_camPos =
                ::Vector3f(loaded.camOrigin.x, loaded.camOrigin.y, loaded.camOrigin.z);
        ::Vector3f fwd(loaded.camTarget.x - loaded.camOrigin.x,
                                     loaded.camTarget.y - loaded.camOrigin.y,
                                     loaded.camTarget.z - loaded.camOrigin.z);
        m_camForward = normalize(fwd);
        m_camUp = loaded.camUp;
    }

    m_sceneLabel->setCaption(fs::path(xmlPath).filename().string());
    if (m_triCountLabel)
        m_triCountLabel->setCaption("Triangles: " +
                                                                std::to_string(loaded.triangles.size()));
    performLayout();
    m_film->clear();
    return true;
    } catch (const std::exception &e) {
        auto dlg = new MessageDialog(this, MessageDialog::Type::Warning,
                                     "Scene load failed", e.what());
        (void)dlg;
        return false;
    }
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
                // Track framebuffer size changes during load
                int fw, fh;
                glfwGetFramebufferSize(win, &fw, &fh);
                if (fw != m_renderWidth || fh != m_renderHeight) {
                    m_renderWidth = fw;
                    m_renderHeight = fh;
                    recreateRenderTargets(fw, fh);
                    performLayout();
                }

                if (m_loadingWindow) {
                    m_loadingWindow->setPosition(nanogui::Vector2i((m_renderWidth - 450) / 2, (m_renderHeight - 180) / 2));
                    m_loadingProgressBar->setValue(futaba::g_optixCompileProgress.load());
                    m_loadingStatusLabel->setCaption(futaba::g_optixCompileStatus.load());
                }

                glViewport(0, 0, m_renderWidth, m_renderHeight);
                glClearColor(0.11f, 0.11f, 0.13f, 1.0f);
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
                }
                performLayout();
            }

            double currentTime = glfwGetTime();
            float deltaTime = (float)(currentTime - lastFrameTime);
            lastFrameTime = currentTime;

            nbFrames++;
            if (currentTime - lastTime >= 1.0) {
                char buf[32];
                snprintf(buf, sizeof(buf), "FPS: %.1f",
                                 (double)nbFrames / (currentTime - lastTime));
                if (m_fpsLabel)
                    m_fpsLabel->setCaption(buf);
                nbFrames = 0;
                lastTime = currentTime;
            }

            float spd = m_moveSpeed * deltaTime;
            if (m_keys[GLFW_KEY_LEFT_SHIFT])
                spd *= 3.f;

            ::Vector3f fwd = m_camForward;
            ::Vector3f up = m_camUp;
            ::Vector3f right = normalize(cross(fwd, up));

            bool moved = false;
            if (m_keys[GLFW_KEY_W]) {
                m_camPos += fwd * spd;
                moved = true;
            }
            if (m_keys[GLFW_KEY_S]) {
                m_camPos += fwd * -spd;
                moved = true;
            }
            if (m_keys[GLFW_KEY_D]) {
                m_camPos += right * spd;
                moved = true;
            }
            if (m_keys[GLFW_KEY_A]) {
                m_camPos += right * -spd;
                moved = true;
            }
            if (m_keys[GLFW_KEY_E]) {
                m_camPos += up * spd;
                moved = true;
            }
            if (m_keys[GLFW_KEY_Q]) {
                m_camPos += up * -spd;
                moved = true;
            }

            // Handle resize interactively in the loop (more reliable than callbacks on
            // some platforms)
            int fw, fh;
            glfwGetFramebufferSize(win, &fw, &fh);
            if (fw != m_renderWidth || fh != m_renderHeight) {
                m_renderWidth = fw;
                m_renderHeight = fh;
                recreateRenderTargets(fw, fh);
                moved = true; // Force camera update
            }

            if (moved)
                updateCamera();

            uchar4 *d_pbo_ptr;
            size_t num_bytes;
            cudaGraphicsMapResources(1, &m_cudaPboResource, 0);
            cudaGraphicsResourceGetMappedPointer((void **)&d_pbo_ptr, &num_bytes,
                                                 m_cudaPboResource);

            uchar4 *d_vis_pbo_ptr = nullptr;
            if (m_showVisualizer && m_cudaPboResourceVis) {
                cudaGraphicsMapResources(1, &m_cudaPboResourceVis, 0);
                cudaGraphicsResourceGetMappedPointer((void **)&d_vis_pbo_ptr, &num_bytes,
                                                     m_cudaPboResourceVis);
            }

            preprocess();

            LaunchParams params = {};
            params.pbo_ptr = d_pbo_ptr;
            params.width = m_renderWidth;
            params.height = m_renderHeight;
            params.camera = m_camera;
            params.scene = m_scene;
            params.max_depth = m_maxDepth;
            params.rr_depth = m_rrDepth;
            params.integrator_mode = m_integratorMode;
            params.tonemapping_mode = m_tonemappingMode;
            params.use_antialiasing = m_useAntialiasing;
            params.phong_light_dir = m_phongLightDir;
            params.phong_ambient = m_phongAmbient;
            params.phong_diffuse = m_phongDiffuse;
            params.phong_specular = m_phongSpecular;
            params.phong_shininess = m_phongShininess;
            params.denoise_active = m_useDenoiser;
            params.path_guiding_mode = m_pathGuidingMode;
            params.light_sampler_type = m_lightSamplerType;

            params.train_active = m_collectTraining ? m_trainManager.getActive() : nullptr;
            params.train_position = m_collectTraining ? m_trainManager.getPosition() : nullptr;
            params.train_normals = m_collectTraining ? m_trainManager.getNormals() : nullptr;
            params.train_wi = m_collectTraining ? m_trainManager.getWi() : nullptr;
            params.train_wo = m_collectTraining ? m_trainManager.getWo() : nullptr;
            params.train_radiance = m_collectTraining ? m_trainManager.getRadiance() : nullptr;
            params.train_material_id = m_collectTraining ? m_trainManager.getMaterialId() : nullptr;

            params.vis_pbo_ptr = d_vis_pbo_ptr;
            params.vis_depth = m_visDepth;
            params.vis_buffer_type = m_visBufferType;
            params.vis_active = m_showVisualizer;

            launch_render(m_film, &m_denoiser, params);

            postprocess();

            cudaGraphicsUnmapResources(1, &m_cudaPboResource, 0);
            if (m_showVisualizer && m_cudaPboResourceVis) {
                cudaGraphicsUnmapResources(1, &m_cudaPboResourceVis, 0);

                // Copy visualizer PBO contents to the visualizer texture
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_glPboVis);
                glBindTexture(GL_TEXTURE_2D, m_glTexVis);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_renderWidth, m_renderHeight,
                                GL_RGBA, GL_UNSIGNED_BYTE, 0);
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            }

            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_glPbo);
            glBindTexture(GL_TEXTURE_2D, m_glTex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_renderWidth, m_renderHeight,
                                            GL_RGBA, GL_UNSIGNED_BYTE, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            glBindFramebuffer(GL_READ_FRAMEBUFFER, m_fbo);
            glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                                         GL_TEXTURE_2D, m_glTex, 0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
            glfwGetFramebufferSize(win, &fw, &fh);
            glBlitFramebuffer(0, 0, m_renderWidth, m_renderHeight, 0, fh, fw, 0,
                                                GL_COLOR_BUFFER_BIT, GL_NEAREST);

            drawContents();
            drawGizmo();
            drawWidgets();
            glfwSwapBuffers(win);
        } catch (const std::exception& e) {
            auto dlg = new MessageDialog(this, MessageDialog::Type::Warning,
                                         "Error", e.what());
            (void)dlg;
        }
    }
}

bool FutabaScreen::keyboardEvent(int key, int scancode, int action,
                                                                 int modifiers) {
    if (Screen::keyboardEvent(key, scancode, action, modifiers))
        return true;
    if (key >= 0 && key < 1024) {
        if (action == GLFW_PRESS)
            m_keys[key] = true;
        else if (action == GLFW_RELEASE)
            m_keys[key] = false;
    }
    return true;
}

bool FutabaScreen::mouseButtonEvent(const nanogui::Vector2i &p, int button,
                                                                        bool down, int modifiers) {
    if (Screen::mouseButtonEvent(p, button, down, modifiers))
        return true;
    if (button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_RIGHT) {
        m_rightMousePressed = down;
        return true;
    }
    return false;
}

bool FutabaScreen::mouseMotionEvent(const nanogui::Vector2i &p,
                                                                        const nanogui::Vector2i &rel, int button,
                                                                        int modifiers) {
    if (Screen::mouseMotionEvent(p, rel, button, modifiers))
        return true;
    if (m_rightMousePressed) {
        float dx = -rel.x() * 0.15f;
        float dy = -rel.y() * 0.15f;

        // Rotate forward around UP by dx
        m_camForward = normalize(::Matrix4f::rotate(m_camUp, dx) * m_camForward);

        // Compute right vector
        ::Vector3f right = normalize(cross(m_camForward, m_camUp));

        // Rotate forward around right by dy
        ::Vector3f newForward =
                normalize(::Matrix4f::rotate(right, dy) * m_camForward);

        // Prevent gimbal lock (don't let forward become parallel to up)
        if (abs(dot(newForward, m_camUp)) < 0.99f) {
            m_camForward = newForward;
        }

        updateCamera();
        return true;
    }
    return false;
}

bool FutabaScreen::scrollEvent(const nanogui::Vector2i &p,
                                                             const nanogui::Vector2f &rel) {
    if (Screen::scrollEvent(p, rel))
        return true;
    m_currentFov -= rel.y() * 2.f;
    if (m_currentFov < kMinFov)
        m_currentFov = kMinFov;
    if (m_currentFov > kMaxFov)
        m_currentFov = kMaxFov;
    if (m_fovSlider)
        m_fovSlider->setValue(fovToSlider(m_currentFov));

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

    // Always update render targets and reset accumulation on resize
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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA,
                             GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenBuffers(1, &m_glPbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_glPbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4 * sizeof(GLubyte),
                             NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&m_cudaPboResource, m_glPbo,
                                                             cudaGraphicsMapFlagsWriteDiscard);

    if (m_film)
        delete m_film;
    m_film = new HDRFilm(width, height);

    if (futaba::getOptixContext()) {
        m_denoiser.resize(width, height);
    }
    m_guiding.resize(width, height);

    // Reallocate training buffers
    m_trainManager.allocate(width, height, m_maxDepth);

    // Reallocate Visualizer resources
    if (m_cudaPboResourceVis != nullptr) {
        cudaGraphicsUnregisterResource(m_cudaPboResourceVis);
        m_cudaPboResourceVis = nullptr;
    }
    if (m_glPboVis != 0) {
        glDeleteBuffers(1, &m_glPboVis);
        m_glPboVis = 0;
    }
    if (m_glTexVis != 0) {
        glDeleteTextures(1, &m_glTexVis);
        m_glTexVis = 0;
    }

    glGenTextures(1, &m_glTexVis);
    glBindTexture(GL_TEXTURE_2D, m_glTexVis);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenBuffers(1, &m_glPboVis);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_glPboVis);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4 * sizeof(GLubyte),
                 NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&m_cudaPboResourceVis, m_glPboVis,
                                 cudaGraphicsMapFlagsWriteDiscard);

    if (m_visImageView) {
        m_visImageView->bindImage(m_glTexVis);
    }
}

void FutabaScreen::updateCamera() {
    int fw, fh;
    glfwGetFramebufferSize(glfwWindow(), &fw, &fh);
    float aspect = (float)fw / (float)fh;

    Point3f pos(m_camPos.x, m_camPos.y, m_camPos.z);
    Point3f target(m_camPos.x + m_camForward.x, m_camPos.y + m_camForward.y,
                                 m_camPos.z + m_camForward.z);

    m_camera.init(pos, target, m_camUp, m_currentFov, aspect,
                  m_currentFocusDistance, m_currentApertureRadius);
    m_film->clear();
}

void FutabaScreen::drawGizmo() {
    NVGcontext *vg = mNVGContext;

    int size = 80;
    int margin = 20;
    // Position using window coordinates (mSize) so it stays top-right regardless
    // of HiDPI
    int centerX = mSize.x() - size / 2 - margin;
    int centerY = size / 2 + margin;

    nvgSave(vg);

    // Background circle
    nvgBeginPath(vg);
    nvgCircle(vg, (float)centerX, (float)centerY, size / 2.0f);
    nvgFillColor(vg, nvgRGBA(40, 40, 40, 150));
    nvgFill(vg);

    auto drawAxis = [&](const ::Vector3f &worldDir, NVGcolor color,
                                            const char *label) {
        // Project world axis to camera plane
        float dx = dot(worldDir, m_camera.right);
        float dy = -dot(worldDir, m_camera.trueUp); // flip Y for screen coordinates

        float len = size * 0.35f;
        float endX = centerX + dx * len;
        float endY = centerY + dy * len;

        // Draw line
        nvgBeginPath(vg);
        nvgMoveTo(vg, (float)centerX, (float)centerY);
        nvgLineTo(vg, endX, endY);
        nvgStrokeColor(vg, color);
        nvgStrokeWidth(vg, 2.0f);
        nvgStroke(vg);

        // Draw label circle
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

    drawAxis(::Vector3f(1, 0, 0), nvgRGBA(255, 50, 50, 255), "X");
    drawAxis(::Vector3f(0, 1, 0), nvgRGBA(50, 255, 50, 255), "Y");
    drawAxis(::Vector3f(0, 0, 1), nvgRGBA(50, 50, 255, 255), "Z");

    nvgRestore(vg);
}

void FutabaScreen::saveTrainingData(const std::string& basePath) {
    m_trainManager.save(basePath, m_renderWidth, m_renderHeight, m_maxDepth);
}

void FutabaScreen::freeTrainingBuffers() {
    m_trainManager.freeBuffers();
}

static cudaTextureObject_t createCudaTexture(const std::string& filename,
                                             std::vector<cudaArray*>& allocatedArrays,
                                             std::vector<cudaTextureObject_t>& allocatedTextures)
{
    int width = 0, height = 0, channels = 0;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 4); // load as RGBA
    if (!data) {
        std::cerr << "Failed to load texture: " << filename << std::endl;
        return 0;
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray* cuArray = nullptr;
    cudaError_t err = cudaMallocArray(&cuArray, &channelDesc, width, height);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc array failed: " << cudaGetErrorString(err) << std::endl;
        stbi_image_free(data);
        return 0;
    }

    err = cudaMemcpy2DToArray(cuArray, 0, 0, data, width * 4 * sizeof(unsigned char), width * 4 * sizeof(unsigned char), height, cudaMemcpyHostToDevice);
    stbi_image_free(data);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy to array failed: " << cudaGetErrorString(err) << std::endl;
        cudaFreeArray(cuArray);
        return 0;
    }

    allocatedArrays.push_back(cuArray);

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    if (err != cudaSuccess) {
        std::cerr << "CUDA create texture object failed: " << cudaGetErrorString(err) << std::endl;
        return 0;
    }

    allocatedTextures.push_back(texObj);
    return texObj;
}

void FutabaScreen::clearTextures() {
    for (auto texObj : m_cudaTextureObjects) {
        if (texObj != 0) {
            cudaDestroyTextureObject(texObj);
        }
    }
    m_cudaTextureObjects.clear();

    for (auto cuArray : m_cudaTextureArrays) {
        if (cuArray != nullptr) {
            cudaFreeArray(cuArray);
        }
    }
    m_cudaTextureArrays.clear();
}

void FutabaScreen::updateVisualizerDropdown() {
    if (!m_visCombo) return;
    
    std::vector<std::string> items;
    if (m_pathGuidingMode == futaba::PATH_GUIDING_NONE) {
        items = {"Active"};
    } else if (m_pathGuidingMode == futaba::PATH_GUIDING_PPG) {
        items = {"Active", "Position", "Outgoing Angle (wo)", "Incoming Radiance"};
    } else if (m_pathGuidingMode == futaba::PATH_GUIDING_NPM) {
        items = {"Active", "Position", "Normals", "Incoming Angle (wi)", "Outgoing Angle (wo)", "Incoming Radiance", "Material ID"};
    }
    
    std::string currentSelectedStr = (m_visBufferType >= 0 && m_visBufferType < (int)g_allBufferNames.size()) 
                                     ? g_allBufferNames[m_visBufferType] : "Active";
    
    m_visCombo->setItems(items);
    
    int newSelectedIndex = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        if (items[i] == currentSelectedStr) {
            newSelectedIndex = (int)i;
            break;
        }
    }
    m_visCombo->setSelectedIndex(newSelectedIndex);
    
    std::string finalSelectedStr = items[newSelectedIndex];
    auto it = std::find(g_allBufferNames.begin(), g_allBufferNames.end(), finalSelectedStr);
    if (it != g_allBufferNames.end()) {
        m_visBufferType = (int)std::distance(g_allBufferNames.begin(), it);
    }

    performLayout();
}

void FutabaScreen::preprocess() {
    m_guiding.preprocess();
}

void FutabaScreen::postprocess() {
    m_guiding.postprocess();
}
