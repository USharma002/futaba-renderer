#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <sstream>

#include <nanogui/nanogui.h>
#include <nanovg_gl.h>
#include <Eigen/Geometry>

#include "types.cuh"
#include "common.cuh"
#include "sampler.cuh"
#include "warp.cuh"

namespace {

enum class EWarpType : int {
    EUniformSquare = 0,
    EConcentricDisk,
    EUniformSphere,
    EUniformHemisphere,
    ECosineHemisphere,
    EBeckmann,
    EWarpTypeCount
};

static const std::string kWarpNames[static_cast<int>(EWarpType::EWarpTypeCount)] = {
    "Uniform Square", "Concentric Disk", "Uniform Sphere", 
    "Uniform Hemisphere", "Cosine Hemisphere", "Beckmann Distribution"
};

enum class EDisplayMode {
    EPointCloud,
    EChiSquareTest
};

enum class EColorMode {
    EParametricUV,
    EGeometryNormals,
    EPdfDensity
};

struct TestMetrics {
    float chi2Value = 0.0f;
    float criticalValue = 0.0f;
    bool passed = false;
};

static float lerpFloat(float a, float b, float t) {
    return a + (b - a) * t;
}

static Vector3f getThermalColor(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return Vector3f(
        0.05f + 0.93f * t,
        0.08f + 0.52f * t,
        0.22f + 0.10f * (1.0f - t)
    );
}

// Inline 2D Riemann grid integrator to calculate analytical expectations for the Chi2 cells
double integrateBin2D(const std::function<double(double, double)>& integrand, 
                      double x0, double x1, double y0, double y1, int resolution = 16) {
    double sum = 0.0;
    double dx = (x1 - x0) / resolution;
    double dy = (y1 - y0) / resolution;
    for (int j = 0; j < resolution; ++j) {
        double y = y0 + (j + 0.5) * dy;
        for (int i = 0; i < resolution; ++i) {
            double x = x0 + (i + 0.5) * dx;
            sum += integrand(x, y);
        }
    }
    return sum * dx * dy;
}

} // namespace

class WarpCanvas : public nanogui::GLCanvas {
public:
    explicit WarpCanvas(Widget *parent) : nanogui::GLCanvas(parent) {
        m_shader.init(
            "warp_points",
            "#version 330\n"
            "uniform mat4 mvp;\n"
            "in vec3 position;\n"
            "in vec4 color;\n"
            "out vec4 fragColor;\n"
            "void main() {\n"
            "    fragColor = color;\n"
            "    gl_Position = mvp * vec4(position, 1.0);\n"
            "    gl_PointSize = 3.0;\n"
            "}\n",
            "#version 330\n"
            "in vec4 fragColor;\n"
            "out vec4 outColor;\n"
            "void main() {\n"
            "    outColor = fragColor;\n"
            "}\n"
        );
        resetCamera();
    }

    ~WarpCanvas() override { m_shader.free(); }

    void resetCamera() {
        m_arcball.setState(Eigen::Quaternionf::Identity());
        m_distance = 2.5f;
    }

    void updatePoints(const std::vector<Vector3f>& positions, const std::vector<nanogui::Color>& colors) {
        m_count = positions.size();
        if (m_count == 0) return;

        nanogui::MatrixXf posMat(3, m_count);
        nanogui::MatrixXf colMat(4, m_count);

        for (size_t i = 0; i < m_count; ++i) {
            posMat.col(static_cast<Eigen::Index>(i)) << positions[i].x, positions[i].y, positions[i].z;
            colMat.col(static_cast<Eigen::Index>(i)) << colors[i].r(), colors[i].g(), colors[i].b(), colors[i].w();
        }

        m_shader.bind();
        m_shader.uploadAttrib("position", posMat);
        m_shader.uploadAttrib("color", colMat);
    }

    bool mouseButtonEvent(const nanogui::Vector2i &p, int button, bool down, int modifiers) override {
        if (button == GLFW_MOUSE_BUTTON_1) {
            m_arcball.button(p - absolutePosition(), down);
            return true;
        }
        return Widget::mouseButtonEvent(p, button, down, modifiers);
    }

    bool mouseMotionEvent(const nanogui::Vector2i &p, const nanogui::Vector2i &rel, int button, int modifiers) override {
        if (button & (1 << GLFW_MOUSE_BUTTON_1)) {
            m_arcball.motion(p - absolutePosition());
            return true;
        }
        return Widget::mouseMotionEvent(p, rel, button, modifiers);
    }

    bool scrollEvent(const nanogui::Vector2i &p, const nanogui::Vector2f &rel) override {
        m_distance = std::clamp(m_distance - rel.y() * 0.15f, 0.8f, 6.0f);
        return true;
    }

    void drawGL() override {
        if (m_count == 0) return;

        glEnable(GL_DEPTH_TEST);

        const float aspect = static_cast<float>(size().x()) / static_cast<float>(size().y());
        nanogui::Matrix4f projection = nanogui::Matrix4f::Zero();
        float f = 1.0f / std::tan((45.0f * static_cast<float>(M_PI) / 180.0f) * 0.5f);
        projection(0, 0) = f / aspect;
        projection(1, 1) = f;
        projection(2, 2) = (20.0f + 0.05f) / (0.05f - 20.0f);
        projection(2, 3) = (2.0f * 20.0f * 0.05f) / (0.05f - 20.0f);
        projection(3, 2) = -1.0f;

        nanogui::Matrix4f view = nanogui::Matrix4f::Identity();
        view(2, 3) = -m_distance;
        view = view * m_arcball.matrix();

        m_shader.bind();
        // Evaluates matrix expression explicitly to fix template deduction errors
        m_shader.setUniform("mvp", (projection * view).eval());
        m_shader.drawArray(GL_POINTS, 0, static_cast<uint32_t>(m_count));

        glDisable(GL_DEPTH_TEST);
    }

private:
    nanogui::GLShader m_shader;
    nanogui::Arcball m_arcball;
    size_t m_count = 0;
    float m_distance = 2.5f;
};

class WarpViewerScreen : public nanogui::Screen {
public:
    WarpViewerScreen(int width, int height)
        : nanogui::Screen(nanogui::Vector2i(width, height), "Importance Sampling Warp Tester") {
        setBackground(nanogui::Color(0.06f, 0.07f, 0.09f, 1.0f));

        m_panel = new nanogui::Window(this, "Control Window");
        m_panel->setPosition(nanogui::Vector2i(15, 15));
        m_panel->setFixedWidth(290);
        m_panel->setLayout(new nanogui::GroupLayout(10, 5, 5, 5));

        new nanogui::Label(m_panel, "Target Function (warp.cuh)", "sans-bold");
        m_warpCombo = new nanogui::ComboBox(m_panel);
        std::vector<std::string> items;
        for (int i = 0; i < static_cast<int>(EWarpType::EWarpTypeCount); ++i) items.push_back(kWarpNames[i]);
        m_warpCombo->setItems(items);
        m_warpCombo->setCallback([this](int idx) { m_warpType = static_cast<EWarpType>(idx); generateAndValidate(); });

        new nanogui::Label(m_panel, "Display Mode", "sans-bold");
        m_modeCombo = new nanogui::ComboBox(m_panel, { "3D Spatial Point Cloud", "Chi-Square Frequency Test" });
        m_modeCombo->setCallback([this](int idx) { m_mode = static_cast<EDisplayMode>(idx); });

        new nanogui::Label(m_panel, "3D Shading Mode", "sans-bold");
        m_colorCombo = new nanogui::ComboBox(m_panel, { "Color by Uniform UV", "Color by Geometry Normals", "Color by PDF Heatmap Density" });
        m_colorCombo->setCallback([this](int idx) { m_colorMode = static_cast<EColorMode>(idx); generateAndValidate(); });

        new nanogui::Label(m_panel, "Sample Allocation Size", "sans-bold");
        auto *sampleSlider = new nanogui::Slider(m_panel);
        sampleSlider->setValue(0.4f);
        sampleSlider->setCallback([this](float val) {
            m_sampleCount = static_cast<int>(std::pow(2.0f, lerpFloat(8.0f, 17.0f, val)));
            generateAndValidate();
        });

        new nanogui::Label(m_panel, "Beckmann Alpha Roughness", "sans-bold");
        auto *alphaSlider = new nanogui::Slider(m_panel);
        alphaSlider->setValue(0.3f);
        alphaSlider->setCallback([this](float val) { m_alpha = std::max(0.01f, val); generateAndValidate(); });

        auto *btnReseed = new nanogui::Button(m_panel, "Reseed Random Generator");
        btnReseed->setCallback([this] { m_seedBase += 13; generateAndValidate(); });

        auto *btnCamera = new nanogui::Button(m_panel, "Reset 3D Camera Focus");
        btnCamera->setCallback([this] { if (m_canvas) m_canvas->resetCamera(); });

        m_statsLabel = new nanogui::Label(m_panel, "", "sans", 13);
        m_statsLabel->setColor(nanogui::Color(0.85f, 0.88f, 0.92f, 1.0f));

        m_canvas = new WarpCanvas(this);
        
        generateAndValidate();
        performLayout();
    }

    void draw(NVGcontext *ctx) override {
        const float screenW = static_cast<float>(width());
        const float screenH = static_cast<float>(height());
        const float canvasX = 320.0f;
        const float margin = 35.0f;
        const float canvasW = screenW - canvasX - margin;
        const float canvasH = screenH - 2.0f * margin;

        if (m_mode == EDisplayMode::EPointCloud) {
            m_canvas->setVisible(true);
            m_canvas->setPosition(nanogui::Vector2i(static_cast<int>(canvasX), static_cast<int>(margin)));
            m_canvas->setSize(nanogui::Vector2i(static_cast<int>(canvasW), static_cast<int>(canvasH)));
        } else {
            m_canvas->setVisible(false);
            renderHistogramGrids(ctx, canvasX, margin, canvasW, canvasH);
        }

        // Draw interface window last to overlay on top layer cleanly
        Screen::draw(ctx);
    }

private:
    void generateAndValidate() {
        futaba::Sampler sampler(m_seedBase);
        std::vector<Vector3f> points;
        std::vector<nanogui::Color> colors;
        points.reserve(m_sampleCount);
        colors.reserve(m_sampleCount);

        m_xres = (m_warpType == EWarpType::EUniformSquare || m_warpType == EWarpType::EConcentricDisk) ? 32 : 64;
        m_yres = 32;
        int totalBins = m_xres * m_yres;
        m_obsFreq.assign(static_cast<size_t>(totalBins), 0.0);
        m_expFreq.assign(static_cast<size_t>(totalBins), 0.0);

        std::vector<float> analyticalDensities(m_sampleCount, 0.0f);
        float maxDensity = 1e-4f;

        for (int i = 0; i < m_sampleCount; ++i) {
            Point2f u = sampler.next2D();
            Vector3f mapped(0.0f);
            float binX = 0.0f, binY = 0.0f;

            // HERE: Executing your actual warp.cuh IMPORTANCE SAMPLING functions directly
            switch (m_warpType) {
                case EWarpType::EUniformSquare: {
                    Point2f res = Warp::squareToUniformSquare(u);
                    mapped = Vector3f(res.x, res.y, 0.0f);
                    binX = res.x; binY = res.y;
                    analyticalDensities[i] = Warp::squareToUniformSquarePdf(res);
                    break;
                }
                case EWarpType::EConcentricDisk: {
                    Point2f res = Warp::squareToConcentricDisk(u);
                    mapped = Vector3f(res.x, res.y, 0.0f);
                    binX = res.x * 0.5f + 0.5f; binY = res.y * 0.5f + 0.5f;
                    analyticalDensities[i] = (res.x*res.x + res.y*res.y <= 1.0f) ? INV_PI : 0.0f;
                    break;
                }
                case EWarpType::EUniformSphere: {
                    mapped = Warp::squareToUniformSphere(u);
                    binX = std::atan2(mapped.y, mapped.x) * INV_PI * 0.5f;
                    if (binX < 0.0f) binX += 1.0f;
                    binY = mapped.z * 0.5f + 0.5f;
                    analyticalDensities[i] = Warp::squareToUniformSpherePdf(mapped);
                    break;
                }
                case EWarpType::EUniformHemisphere: {
                    mapped = Warp::squareToUniformHemisphere(u);
                    binX = std::atan2(mapped.y, mapped.x) * INV_PI * 0.5f;
                    if (binX < 0.0f) binX += 1.0f;
                    binY = mapped.z;
                    analyticalDensities[i] = Warp::squareToUniformHemispherePdf(mapped);
                    break;
                }
                case EWarpType::ECosineHemisphere: {
                    mapped = Warp::squareToCosineHemisphere(u);
                    binX = std::atan2(mapped.y, mapped.x) * INV_PI * 0.5f;
                    if (binX < 0.0f) binX += 1.0f;
                    binY = mapped.z;
                    analyticalDensities[i] = Warp::squareToCosineHemispherePdf(mapped);
                    break;
                }
                case EWarpType::EBeckmann: {
                    mapped = Warp::squareToBeckmann(u, m_alpha);
                    binX = std::atan2(mapped.y, mapped.x) * INV_PI * 0.5f;
                    if (binX < 0.0f) binX += 1.0f;
                    binY = mapped.z;
                    analyticalDensities[i] = Warp::squareToBeckmannPdf(mapped, m_alpha);
                    break;
                }
                default: break;
            }

            maxDensity = std::max(maxDensity, analyticalDensities[i]);
            points.push_back(mapped);

            // THIS IS NOT SAMPLING. This projects the 3D direction vector onto a flat 2D cell grid for histogramming
            int bx = std::min(m_xres - 1, std::max(0, static_cast<int>(binX * m_xres)));
            int by = std::min(m_yres - 1, std::max(0, static_cast<int>(binY * m_yres)));
            m_obsFreq[static_cast<size_t>(by * m_xres + bx)] += 1.0;
        }

        // Color assigning pass
        for (int i = 0; i < m_sampleCount; ++i) {
            if (m_colorMode == EColorMode::EParametricUV) {
                colors.push_back(nanogui::Color(points[i].x * 0.5f + 0.5f, points[i].y * 0.5f + 0.5f, 0.6f, 0.8f));
            } else if (m_colorMode == EColorMode::EGeometryNormals) {
                Vector3f n = normalize(points[i]);
                colors.push_back(nanogui::Color(std::abs(n.x), std::abs(n.y), std::abs(n.z), 0.8f));
            } else {
                Vector3f rgb = getThermalColor(analyticalDensities[i] / maxDensity);
                colors.push_back(nanogui::Color(rgb.x, rgb.y, rgb.z, 0.8f));
            }
        }

        m_canvas->updatePoints(points, colors);

        // HERE: Evaluation of your analytical Pdf() methods to build testing expectation arrays
        auto integrand = [this](double x, double y) -> double {
            if (m_warpType == EWarpType::EUniformSquare) {
                return Warp::squareToUniformSquarePdf(Point2f(static_cast<float>(x), static_cast<float>(y)));
            } 
            else if (m_warpType == EWarpType::EConcentricDisk) {
                float dx = static_cast<float>(x * 2.0 - 1.0);
                float dy = static_cast<float>(y * 2.0 - 1.0);
                return (dx*dx + dy*dy <= 1.0f) ? INV_PI : 0.0;
            } 
            else {
                double phi = x * 2.0 * M_PI;
                double z = (m_warpType == EWarpType::EUniformSphere) ? (y * 2.0 - 1.0) : y;
                double sinTheta = std::sqrt(std::max(0.0, 1.0 - z * z));
                Vector3f v(static_cast<float>(sinTheta * std::cos(phi)), 
                           static_cast<float>(sinTheta * std::sin(phi)), 
                           static_cast<float>(z));

                if (m_warpType == EWarpType::EUniformSphere) return Warp::squareToUniformSpherePdf(v);
                if (m_warpType == EWarpType::EUniformHemisphere) return Warp::squareToUniformHemispherePdf(v);
                if (m_warpType == EWarpType::ECosineHemisphere) return Warp::squareToCosineHemispherePdf(v);
                if (m_warpType == EWarpType::EBeckmann) return Warp::squareToBeckmannPdf(v, m_alpha);
                return 0.0;
            }
        };

        double analyticalScale = m_sampleCount;
        if (m_warpType == EWarpType::EConcentricDisk) analyticalScale *= 4.0;
        else if (m_warpType == EWarpType::EUniformSphere) analyticalScale *= 4.0 * M_PI;
        else if (m_warpType != EWarpType::EUniformSquare) analyticalScale *= 2.0 * M_PI;

        for (int by = 0; by < m_yres; ++by) {
            double y0 = static_cast<double>(by) / m_yres;
            double y1 = static_cast<double>(by + 1) / m_yres;
            for (int bx = 0; bx < m_xres; ++bx) {
                double x0 = static_cast<double>(bx) / m_xres;
                double x1 = static_cast<double>(bx + 1) / m_xres;

                m_expFreq[static_cast<size_t>(by * m_xres + bx)] = integrateBin2D(integrand, x0, x1, y0, y1) * analyticalScale;
            }
        }

        // Compute Chi2 statistics
        m_test.chi2Value = 0.0f;
        int degreesOfFreedom = 0;
        for (int i = 0; i < totalBins; ++i) {
            double E = m_expFreq[static_cast<size_t>(i)];
            if (E > 1e-3) {
                double O = m_obsFreq[static_cast<size_t>(i)];
                m_test.chi2Value += static_cast<float>(((O - E) * O - E) / E);
                degreesOfFreedom++;
            }
        }
        degreesOfFreedom = std::max(1, degreesOfFreedom - 1);
        m_test.criticalValue = static_cast<float>(degreesOfFreedom + 2.33f * std::sqrt(2.0f * degreesOfFreedom));
        m_test.passed = (m_test.chi2Value <= m_test.criticalValue);

        std::string statSummary = 
            "Allocated Points: " + std::to_string(m_sampleCount) + "\n" +
            "Bin Layout: " + std::to_string(m_xres) + "x" + std::to_string(m_yres) + "\n" +
            "Degrees of Freedom: " + std::to_string(degreesOfFreedom) + "\n" +
            "Chi2 Stat: " + std::to_string(m_test.chi2Value) + "\n" +
            "Critical Bound: " + std::to_string(m_test.criticalValue) + "\n\n" +
            "Hypothesis Status:\n" + (m_test.passed ? "[ PASSED ]" : "[ FAILED ]");
        
        m_statsLabel->setCaption(statSummary);
        m_panel->performLayout(screen()->nvgContext());
    }

    void renderHistogramGrids(NVGcontext *ctx, float x, float y, float w, float h) {
        float plotH = (h - 40.0f) * 0.5f;

        double maxObs = 1e-4, maxExp = 1e-4;
        for (double v : m_obsFreq) maxObs = std::max(maxObs, v);
        for (double v : m_expFreq) maxExp = std::max(maxExp, v);

        auto drawGrid = [&](float startY, const std::vector<double>& data, double maxVal, const std::string& title) {
            nvgBeginPath(ctx);
            nvgRoundedRect(ctx, x, startY, w, plotH, 6.0f);
            nvgFillColor(ctx, nvgRGBAf(0.03f, 0.04f, 0.05f, 1.0f));
            nvgFill(ctx);

            float cellW = w / static_cast<float>(m_xres);
            float cellH = (plotH - 30.0f) / static_cast<float>(m_yres);

            for (int by = 0; by < m_yres; ++by) {
                for (int bx = 0; bx < m_xres; ++bx) {
                    double val = data[static_cast<size_t>(by * m_xres + bx)];
                    float cx = x + static_cast<float>(bx) * cellW;
                    float cy = startY + plotH - 5.0f - static_cast<float>(by + 1) * cellH;

                    Vector3f rgb = getThermalColor(static_cast<float>(val / maxVal));
                    nvgBeginPath(ctx);
                    nvgRect(ctx, cx, cy, cellW + 0.3f, cellH + 0.3f);
                    nvgFillColor(ctx, nvgRGBAf(rgb.x, rgb.y, rgb.z, 1.0f));
                    nvgFill(ctx);
                }
            }

            nvgFontFace(ctx, "sans-bold");
            nvgFontSize(ctx, 14.0f);
            nvgFillColor(ctx, nvgRGBAf(0.85f, 0.88f, 0.92f, 0.95f));
            nvgText(ctx, x + 12.0f, startY + 20.0f, title.c_str(), nullptr);
        };

        drawGrid(y, m_obsFreq, maxObs, "Empirical Monte Carlo Sampled Histogram Density (From warp.cuh)");
        drawGrid(y + plotH + 40.0f, m_expFreq, maxExp, "Analytical Mathematically Integrated Expected PDF (From warp.cuh)");

        float bannerY = y + plotH + 12.0f;
        nvgBeginPath(ctx);
        nvgRect(ctx, x, bannerY, w, 18.0f);
        nvgFillColor(ctx, m_test.passed ? nvgRGBAf(0.12f, 0.45f, 0.18f, 0.85f) : nvgRGBAf(0.55f, 0.12f, 0.15f, 0.85f));
        nvgFill(ctx);

        nvgFontSize(ctx, 11.0f);
        nvgTextAlign(ctx, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);
        nvgFillColor(ctx, nvgRGBAf(1.0f, 1.0f, 1.0f, 1.0f));
        std::string bannerTxt = m_test.passed ? "CHI-SQUARE HYPOTHESIS PASS: Generated points match your analytical PDF methods perfectly."
                                              : "CHI-SQUARE HYPOTHESIS CRITICAL FAILURE: Parity between sampling generation and PDF broke.";
        nvgText(ctx, x + w * 0.5f, bannerY + 9.0f, bannerTxt.c_str(), nullptr);
    }

private:
    nanogui::Window *m_panel = nullptr;
    nanogui::Label *m_statsLabel = nullptr;
    nanogui::ComboBox *m_warpCombo = nullptr;
    nanogui::ComboBox *m_modeCombo = nullptr;
    nanogui::ComboBox *m_colorCombo = nullptr;
    WarpCanvas *m_canvas = nullptr;

    EWarpType m_warpType = EWarpType::EUniformSquare;
    EDisplayMode m_mode = EDisplayMode::EPointCloud;
    EColorMode m_colorMode = EColorMode::EParametricUV;
    
    int m_sampleCount = 16384;
    float m_alpha = 0.3f;
    unsigned m_seedBase = 42;

    int m_xres = 32;
    int m_yres = 32;
    std::vector<double> m_obsFreq;
    std::vector<double> m_expFreq;
    TestMetrics m_test;
};

int main() {
    try {
        nanogui::init();
        {
            nanogui::ref<WarpViewerScreen> screen = new WarpViewerScreen(1280, 760);
            screen->drawAll();
            screen->setVisible(true);
            nanogui::mainloop();
        }
        nanogui::shutdown();
    } catch (const std::exception &e) {
        std::cerr << "Fatal execution error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}