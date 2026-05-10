#include "gui.h"
#include "bitmap.h"
#include "hdrfilm.cuh"
#include "renderer.h"
#include "scene_loader.h"
#include <filesystem>
#include <iostream>

using namespace nanogui;
using namespace futaba;
namespace fs = std::filesystem;

static constexpr float kMinFov = 5.f;
static constexpr float kMaxFov = 120.f;

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

FutabaScreen::FutabaScreen(int width, int height)
        : Screen(nanogui::Vector2i(width, height), "Futaba Renderer") {
    // Initialize render size from framebuffer
    glfwGetFramebufferSize(glfwWindow(), &m_renderWidth, &m_renderHeight);

    Window *window = new Window(this, "Settings");
    window->setPosition(nanogui::Vector2i(15, 15));
    window->setLayout(new GroupLayout(10, 5, 5, 5));

    Widget *btnPanel = new Widget(window);
    btnPanel->setLayout(
            new BoxLayout(Orientation::Horizontal, Alignment::Fill, 3, 3));

    Button *btnReset = new Button(btnPanel, "Reset");
    btnReset->setCallback([this] { m_film->clear(); });

    Button *btnSave = new Button(btnPanel, "Save EXR");
    btnSave->setCallback([this] {
        std::string path = file_dialog({{"exr", "OpenEXR"}}, true);
        if (!path.empty()) {
            Bitmap *bmp = m_film->toBitmap();
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
    ComboBox *integratorCombo =
            new ComboBox(window, {"Path", "Normals", "Heatmap"});
    integratorCombo->setSelectedIndex((int)m_integratorMode);
    integratorCombo->setCallback([this](int index) {
        m_integratorMode = index;
        m_film->clear();
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

    new Label(window, "FOV", "sans-bold");
    m_fovSlider = new Slider(window);
    m_fovSlider->setValue(fovToSlider(m_currentFov));
    m_fovSlider->setCallback([this](float value) {
        m_currentFov = sliderToFov(value);
        m_camera.setFov(m_currentFov);
        m_film->clear();
    });
    
    m_fpsLabel = new Label(window, "FPS: 0.0");
    m_fpsLabel->setFont("sans-bold");

    new Label(window, "Depth", "sans-bold");
    Widget *maxDepthPanel = new Widget(window);
    maxDepthPanel->setLayout(
            new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 10));
    Slider *maxDepthSlider = new Slider(maxDepthPanel);
    maxDepthSlider->setValue(m_maxDepth / 32.f);
    maxDepthSlider->setFixedWidth(100);
    Label *maxDepthVal = new Label(maxDepthPanel, std::to_string(m_maxDepth));
    maxDepthSlider->setCallback([this, maxDepthVal](float value) {
        m_maxDepth = std::max(1, (int)(value * 32.f));
        maxDepthVal->setCaption(std::to_string(m_maxDepth));
        m_film->clear();
    });

    new Label(window, "RR", "sans-bold");
    Widget *rrDepthPanel = new Widget(window);
    rrDepthPanel->setLayout(
            new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 10));
    Slider *rrDepthSlider = new Slider(rrDepthPanel);
    rrDepthSlider->setValue(m_rrDepth / 16.f);
    rrDepthSlider->setFixedWidth(100);
    Label *rrDepthVal = new Label(rrDepthPanel, std::to_string(m_rrDepth));
    rrDepthSlider->setCallback([this, rrDepthVal](float value) {
        m_rrDepth = std::max(1, (int)(value * 16.f));
        rrDepthVal->setCaption(std::to_string(m_rrDepth));
        m_film->clear();
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

    if (m_triCountLabel)
        m_triCountLabel->setCaption("Triangles: " +
                                                                std::to_string(m_scene.triangleCount));

    // Ensure camera is fully initialized with correct aspect ratio
    updateCamera();
}

FutabaScreen::~FutabaScreen() {
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
    m_scene.setTriangles(loaded.triangles.data(),
                                             (uint32_t)loaded.triangles.size());
    m_scene.setMaterials(loaded.materials.data(),
                                             (uint32_t)loaded.materials.size());
    
    // Convert and upload mesh instances
    std::vector<futaba::MeshInstanceGPU> meshGPU;
    for (const auto& mesh : loaded.meshes) {
        futaba::MeshInstanceGPU m;
        m.triangleStart = mesh.triangleStart;
        m.triangleCount = mesh.triangleCount;
        m.emitterId = mesh.emitterId;
        meshGPU.push_back(m);
    }
    m_scene.setMeshes(meshGPU.data(), (uint32_t)meshGPU.size());

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
    m_scene.setEmitters(emittersGPU.data(), (uint32_t)emittersGPU.size());
    
    m_scene.use_vertex_normals = m_useVertexNormals;

    if (loaded.hasCamera) {
        int fw, fh;
        glfwGetFramebufferSize(glfwWindow(), &fw, &fh);
        float currentAspect = (float)fw / (float)fh;

        m_camera.init(loaded.camOrigin, loaded.camTarget, loaded.camUp,
                                    loaded.camFov, currentAspect);
        m_currentFov = loaded.camFov;
        if (m_fovSlider)
            m_fovSlider->setValue(fovToSlider(m_currentFov));
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
}

void FutabaScreen::renderLoop() {
    GLFWwindow *win = glfwWindow();
    double lastTime = glfwGetTime(), lastFrameTime = lastTime;
    int nbFrames = 0;

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

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

        launch_render(d_pbo_ptr, m_film, m_renderWidth, m_renderHeight, m_camera,
                                    m_scene, m_maxDepth, m_rrDepth, m_integratorMode,
                                    m_useAntialiasing);

        cudaGraphicsUnmapResources(1, &m_cudaPboResource, 0);

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
}

void FutabaScreen::updateCamera() {
    int fw, fh;
    glfwGetFramebufferSize(glfwWindow(), &fw, &fh);
    float aspect = (float)fw / (float)fh;

    Point3f pos(m_camPos.x, m_camPos.y, m_camPos.z);
    Point3f target(m_camPos.x + m_camForward.x, m_camPos.y + m_camForward.y,
                                 m_camPos.z + m_camForward.z);

    m_camera.init(pos, target, m_camUp, m_currentFov, aspect);
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
