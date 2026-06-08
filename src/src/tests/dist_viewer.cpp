#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <nanogui/nanogui.h>

#include "types.cuh"
#include "common.cuh"
#include "distribution.cuh"
#include "sampler.cuh"

namespace {

enum class EDistDimension {
    E1D,
    E2D
};

static float lerpFloat(float a, float b, float t) {
    return a + (b - a) * t;
}

// Heatmap color generator: Slate blue/black -> deep purple -> vibrant orange -> yellow/white
static NVGcolor getHeatmapColor(float t, float alpha = 1.0f) {
    t = std::clamp(t, 0.0f, 1.0f);
    return nvgRGBAf(
        lerpFloat(0.06f, 0.98f, t),
        lerpFloat(0.10f, 0.62f, t),
        lerpFloat(0.18f, 0.12f, t),
        alpha
    );
}

std::vector<float> generateRandomPDFData(int size, unsigned seed) {
    std::vector<float> values(static_cast<size_t>(size), 0.0f);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.1f, 1.0f);
    
    // Mix low-frequency sines, high-frequency sines, and random spikes
    float freq1 = lerpFloat(1.0f, 4.0f, dist(rng));
    float freq2 = lerpFloat(5.0f, 12.0f, dist(rng));
    
    for (int i = 0; i < size; ++i) {
        float x = static_cast<float>(i) / static_cast<float>(size);
        float base = std::sin(x * freq1 * static_cast<float>(M_PI)) * 0.5f + 0.5f;
        float high = std::sin(x * freq2 * static_cast<float>(M_PI)) * 0.25f + 0.25f;
        
        values[static_cast<size_t>(i)] = std::max(0.01f, base + high);
    }
    
    // Add localized high-energy lobes (simulating sun disks or caustic pathways)
    for (int lobe = 0; lobe < 2; ++lobe) {
        int center = static_cast<int>(dist(rng) * static_cast<float>(size));
        int radius = static_cast<int>(0.08f * static_cast<float>(size)) + 2;
        float intensity = dist(rng) * 3.0f;
        
        for (int i = std::max(0, center - radius); i < std::min(size, center + radius); ++i) {
            float d = static_cast<float>(std::abs(i - center)) / static_cast<float>(radius);
            values[static_cast<size_t>(i)] += intensity * (1.0f - d * d);
        }
    }
    return values;
}

} // namespace

class DistributionViewerScreen : public nanogui::Screen {
public:
    DistributionViewerScreen(int width, int height)
        : nanogui::Screen(nanogui::Vector2i(width, height), "Distribution Sampling Diagnostic Visualizer") {
        setBackground(nanogui::Color(0.06f, 0.07f, 0.09f, 1.0f));

        m_panel = new nanogui::Window(this, "Control Panel");
        m_panel->setPosition(nanogui::Vector2i(15, 15));
        m_panel->setFixedWidth(280);
        m_panel->setLayout(new nanogui::GroupLayout(12, 6, 6, 6));

        new nanogui::Label(m_panel, "Dimension Layout", "sans-bold");
        m_dimCombo = new nanogui::ComboBox(m_panel, { "1D Step Distribution", "2D Grid Distribution" });
        m_dimCombo->setCallback([this](int index) {
            m_dim = static_cast<EDistDimension>(index);
            rebuildAndSample();
        });

        new nanogui::Label(m_panel, "Sample Allocation Size", "sans-bold");
        auto *sampleBox = new nanogui::IntBox<int>(m_panel, m_sampleCount);
        sampleBox->setMinMaxValues(10, 500000);
        sampleBox->setSpinnable(true);
        sampleBox->setCallback([this](int val) { m_sampleCount = val; runSamplingOnly(); });

        new nanogui::Label(m_panel, "1D / 2D Grid Resolution", "sans-bold");
        auto *resBox = new nanogui::IntBox<int>(m_panel, m_resolution);
        resBox->setMinMaxValues(8, 128);
        resBox->setSpinnable(true);
        resBox->setCallback([this](int val) { m_resolution = val; rebuildAndSample(); });

        new nanogui::Label(m_panel, "Actions", "sans-bold");
        auto *btnRegen = new nanogui::Button(m_panel, "Regenerate Random PDF");
        btnRegen->setCallback([this] { m_pdfSeed++; rebuildAndSample(); });

        auto *btnResample = new nanogui::Button(m_panel, "Resample Current PDF");
        btnResample->setCallback([this] { m_sampleSeed++; runSamplingOnly(); });

        rebuildAndSample();
        performLayout();
    }

    void draw(NVGcontext *ctx) override {
        const float screenW = static_cast<float>(width());
        const float screenH = static_cast<float>(height());
        
        // Fixed coordinates keep dragging the UI panel from layout-warping the plots
        const float canvasAreaX = 310.0f;
        const float margin = 40.0f;
        
        float canvasAreaW = screenW - canvasAreaX - margin;
        float canvasAreaH = screenH - 2.0f * margin;

        if (canvasAreaW >= 100.0f && canvasAreaH >= 100.0f) {
            if (m_dim == EDistDimension::E1D) {
                render1DHistograms(ctx, canvasAreaX, margin, canvasAreaW, canvasAreaH);
            } else {
                render2DGrids(ctx, canvasAreaX, margin, canvasAreaW, canvasAreaH);
            }
        }

        // Draw the window interface last so it sits cleanly on the top layer
        Screen::draw(ctx);
    }

private:
    void rebuildAndSample() {
        if (m_dim == EDistDimension::E1D) {
            m_rawPdf1D = generateRandomPDFData(m_resolution, m_pdfSeed);
            m_dist1D.build(m_rawPdf1D);
        } else {
            m_rawPdf2D = generateRandomPDFData(m_resolution * m_resolution, m_pdfSeed);
            m_dist2D.build(m_rawPdf2D, m_resolution, m_resolution);
        }
        runSamplingOnly();
    }

    void runSamplingOnly() {
        futaba::Sampler sampler(m_sampleSeed);

        if (m_dim == EDistDimension::E1D) {
            m_sampled1D.assign(static_cast<size_t>(m_resolution), 0.0f);
            
            for (int i = 0; i < m_sampleCount; ++i) {
                float u = sampler.next1D();
                int index = -1;
                float pdf = 0.0f, du = 0.0f;
                m_dist1D.sample(u, index, pdf, du);
                if (index >= 0 && index < m_resolution) {
                    m_sampled1D[static_cast<size_t>(index)] += 1.0f;
                }
            }
            for (auto &val : m_sampled1D) {
                val = (val / static_cast<float>(m_sampleCount)) * static_cast<float>(m_resolution);
            }
        } 
        else {
            m_sampled2D.assign(static_cast<size_t>(m_resolution * m_resolution), 0.0f);
            
            for (int i = 0; i < m_sampleCount; ++i) {
                Point2f samp = sampler.next2D();
                int iu = -1, iv = -1;
                float pdf_v = 0.0f, pdf_u = 0.0f, du = 0.0f, dv = 0.0f;
                
                m_dist2D.marginal.sample(samp.y, iv, pdf_v, dv);
                if (iv >= 0 && iv < m_resolution) {
                    m_dist2D.cond[static_cast<size_t>(iv)].sample(samp.x, iu, pdf_u, du);
                    if (iu >= 0 && iu < m_resolution) {
                        m_sampled2D[static_cast<size_t>(iv * m_resolution + iu)] += 1.0f;
                    }
                }
            }
            float scale = static_cast<float>(m_resolution * m_resolution) / static_cast<float>(m_sampleCount);
            for (auto &val : m_sampled2D) {
                val *= scale;
            }
        }
    }

    void render1DHistograms(NVGcontext *ctx, float x, float y, float w, float h) {
        float subPlotH = (h - 40.0f) * 0.5f;
        float analyticalSum = m_dist1D.funcSum;

        // Calculate analytical peak scale independently without vector mutations
        float maxAnalytical = 1e-4f;
        for (float v : m_rawPdf1D) {
            float normV = (analyticalSum > 0.0f) ? (v / analyticalSum) * static_cast<float>(m_resolution) : v;
            maxAnalytical = std::max(maxAnalytical, normV);
        }

        // Calculate empirical peak scale independently
        float maxEmpirical = 1e-4f;
        for (float v : m_sampled1D) maxEmpirical = std::max(maxEmpirical, v);

        float binW = w / static_cast<float>(m_resolution);

        // Subplot 1: Analytical Reference (Strictly scaled to its own max value)
        nvgBeginPath(ctx);
        nvgRoundedRect(ctx, x, y, w, subPlotH, 6.0f);
        nvgFillColor(ctx, nvgRGBAf(0.03f, 0.04f, 0.05f, 1.0f));
        nvgFill(ctx);

        for (size_t i = 0; i < m_rawPdf1D.size(); ++i) {
            float rawVal = m_rawPdf1D[i];
            float normVal = (analyticalSum > 0.0f) ? (rawVal / analyticalSum) * static_cast<float>(m_resolution) : rawVal;
            float barH = (normVal / maxAnalytical) * (subPlotH - 35.0f);
            float barX = x + static_cast<float>(i) * binW;
            float barY = y + subPlotH - barH - 10.0f;

            nvgBeginPath(ctx);
            nvgRect(ctx, barX, barY, binW - 0.5f, barH);
            nvgFillColor(ctx, getHeatmapColor(normVal / maxAnalytical));
            nvgFill(ctx);
        }
        nvgFontFace(ctx, "sans-bold");
        nvgFontSize(ctx, 14.0f);
        nvgFillColor(ctx, nvgRGBAf(0.8f, 0.82f, 0.85f, 0.9f));
        nvgText(ctx, x + 12.0f, y + 22.0f, "Analytical Reference PDF Structure (Normalized)", nullptr);

        // Subplot 2: Empirical Sampled Result (Strictly scaled to its own max value)
        float plot2Y = y + subPlotH + 40.0f;
        nvgBeginPath(ctx);
        nvgRoundedRect(ctx, x, plot2Y, w, subPlotH, 6.0f);
        nvgFillColor(ctx, nvgRGBAf(0.03f, 0.04f, 0.05f, 1.0f));
        nvgFill(ctx);

        for (size_t i = 0; i < m_sampled1D.size(); ++i) {
            float val = m_sampled1D[i];
            float barH = (val / maxEmpirical) * (subPlotH - 35.0f);
            float barX = x + static_cast<float>(i) * binW;
            float barY = plot2Y + subPlotH - barH - 10.0f;

            nvgBeginPath(ctx);
            nvgRect(ctx, barX, barY, binW - 0.5f, barH);
            nvgFillColor(ctx, getHeatmapColor(val / maxEmpirical));
            nvgFill(ctx);
        }
        nvgText(ctx, x + 12.0f, plot2Y + 22.0f, "Empirical Sampled Frequency Histogram (Inversion Processing)", nullptr);
    }

    void render2DGrids(NVGcontext *ctx, float x, float y, float w, float h) {
        float size = std::min((w - 40.0f) * 0.5f, h - 40.0f);
        size = std::max(size, 100.0f);

        float grid1X = x;
        float grid2X = x + size + 40.0f;

        float maxAnalytical = 1e-4f;
        for (float v : m_rawPdf2D) maxAnalytical = std::max(maxAnalytical, v);
        float maxEmpirical = 1e-4f;
        for (float v : m_sampled2D) maxEmpirical = std::max(maxEmpirical, v);

        auto drawGrid = [&](float startX, const std::vector<float> &data, float maxVal, const std::string &title) {
            float cellS = size / static_cast<float>(m_resolution);
            
            nvgBeginPath(ctx);
            nvgRoundedRect(ctx, startX, y + 30.0f, size, size, 6.0f);
            nvgFillColor(ctx, nvgRGBAf(0.03f, 0.04f, 0.05f, 1.0f));
            nvgFill(ctx);

            for (int iv = 0; iv < m_resolution; ++iv) {
                for (int iu = 0; iu < m_resolution; ++iu) {
                    float val = data[static_cast<size_t>(iv * m_resolution + iu)];
                    float cx = startX + static_cast<float>(iu) * cellS;
                    float cy = (y + 30.0f) + size - static_cast<float>(iv + 1) * cellS;

                    nvgBeginPath(ctx);
                    nvgRect(ctx, cx, cy, cellS + 0.3f, cellS + 0.3f);
                    nvgFillColor(ctx, getHeatmapColor(val / maxVal));
                    nvgFill(ctx);
                }
            }

            nvgFontFace(ctx, "sans-bold");
            nvgFontSize(ctx, 14.0f);
            nvgFillColor(ctx, nvgRGBAf(0.8f, 0.82f, 0.85f, 0.9f));
            nvgText(ctx, startX, y + 18.0f, title.c_str(), nullptr);
        };

        drawGrid(grid1X, m_rawPdf2D, maxAnalytical, "Target Joint Distribution 2D PDF");
        drawGrid(grid2X, m_sampled2D, maxEmpirical, "Inversion Sampled Joint Histogram");
    }

private:
    nanogui::Window *m_panel = nullptr;
    nanogui::ComboBox *m_dimCombo = nullptr;

    EDistDimension m_dim = EDistDimension::E1D;
    int m_sampleCount = 100000;
    int m_resolution = 32;

    unsigned m_pdfSeed = 42;
    unsigned m_sampleSeed = 1337;

    std::vector<float> m_rawPdf1D;
    std::vector<float> m_sampled1D;
    futaba::Distribution1D m_dist1D;

    std::vector<float> m_rawPdf2D;
    std::vector<float> m_sampled2D;
    futaba::Distribution2D m_dist2D;
};

int main() {
    try {
        nanogui::init();
        {
            nanogui::ref<DistributionViewerScreen> screen = new DistributionViewerScreen(1280, 720);
            screen->drawAll();
            screen->setVisible(true);
            nanogui::mainloop();
        }
        nanogui::shutdown();
    } catch (const std::exception &e) {
        std::cerr << "Fatal diagnostic error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}