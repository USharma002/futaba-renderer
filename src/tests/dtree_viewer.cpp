#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <nanogui/nanogui.h>

#include "ppg/dtree.cuh"
#include "sampler.cuh"

// Bring the newly isolated ppg types into scope for nanoGUI
using namespace futaba;

namespace {

enum class ELightingEnv {
    ESingleSpot,
    EDualLobes,
    EComplexCross
};

enum class ESamplingStrategy {
    EUniform,
    EGuided
};

enum class EViewMode {
    ESamplingTree,
    EBuildingTree
};

struct TreeMetrics {
    size_t leafCount = 0;
    int depth = 1;
    float totalWeight = 0.0f;
    float maxLeafWeight = 0.0f;
};

struct ScreenRect {
    float x = 0.0f, y = 0.0f, w = 0.0f, h = 0.0f;
};

static float lerpFloat(float a, float b, float t) {
    return a + (b - a) * t;
}

static ScreenRect toScreenRect(float originX, float originY, float size,
                               float canvasX, float canvasY, float canvasSize) {
    ScreenRect rect;
    rect.x = canvasX + originX * canvasSize;
    rect.y = canvasY + (1.0f - originY - size) * canvasSize;
    rect.w = size * canvasSize;
    rect.h = size * canvasSize;
    return rect;
}

static void drawRect(NVGcontext *ctx, const ScreenRect &rect, const NVGcolor &fill,
                     const NVGcolor &stroke, float strokeWidth) {
    nvgBeginPath(ctx);
    nvgRect(ctx, rect.x, rect.y, rect.w, rect.h);
    nvgFillColor(ctx, fill);
    nvgFill(ctx);

    nvgBeginPath(ctx);
    nvgRect(ctx, rect.x, rect.y, rect.w, rect.h);
    nvgStrokeColor(ctx, stroke);
    nvgStrokeWidth(ctx, strokeWidth);
    nvgStroke(ctx);
}

float evaluateTargetRadiance(const Vector3f &d, ELightingEnv env) {
    float len = std::sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
    if (len < 1e-5f) return 0.0f;
    Vector3f nd(d.x / len, d.y / len, d.z / len);

    if (env == ELightingEnv::ESingleSpot) {
        Vector3f lightDir = Vector3f(0.3f, 0.6f, 0.74f);
        float dot = nd.x*lightDir.x + nd.y*lightDir.y + nd.z*lightDir.z;
        return (dot > 0.93f) ? 45.0f : 0.05f;
    } 
    else if (env == ELightingEnv::EDualLobes) {
        Vector3f l1(0.5f, 0.5f, 0.707f);
        Vector3f l2(-0.6f, -0.4f, 0.2f);
        float dot1 = nd.x*l1.x + nd.y*l1.y + nd.z*l1.z;
        float dot2 = nd.x*l2.x + nd.y*l2.y + nd.z*l2.z;
        
        float rad = 0.0f;
        if (dot1 > 0.96f) rad += 60.0f;
        if (dot2 > 0.80f) rad += 4.0f * std::pow((dot2 - 0.80f) / 0.20f, 2.0f);
        return rad + 0.02f;
    } 
    else {
        float pattern = std::sin(nd.x * 5.0f) * std::cos(nd.y * 5.0f) * nd.z;
        return (pattern > 0.2f) ? 15.0f : 0.01f;
    }
}

void collectMetrics(const QuadTreeNode* nodes, size_t numNodes, size_t nodeIndex, int depthLevel, TreeMetrics &metrics) {
    if (numNodes == 0 || nodes == nullptr) return;
    metrics.depth = std::max(metrics.depth, depthLevel);
    const QuadTreeNode &node = nodes[nodeIndex];

    for (int q = 0; q < 4; ++q) {
        float weight = node.sum(q);
        if (node.isLeaf(q)) {
            metrics.totalWeight += weight;
            metrics.maxLeafWeight = std::max(metrics.maxLeafWeight, weight);
            ++metrics.leafCount;
        } else {
            if (node.child(q) < numNodes) {
                collectMetrics(nodes, numNodes, node.child(q), depthLevel + 1, metrics);
            }
        }
    }
}

void drawTreeCell(NVGcontext *ctx, const QuadTreeNode* nodes, size_t numNodes, size_t nodeIndex,
                  float originX, float originY, float size,
                  float canvasX, float canvasY, float canvasSize,
                  float maxLeafWeight, int depthLevel) {
    if (numNodes == 0 || nodes == nullptr) return;
    const QuadTreeNode &node = nodes[nodeIndex];
    const float childSize = size * 0.5f;

    for (int q = 0; q < 4; ++q) {
        float childOriginX = originX + ((q & 1) ? childSize : 0.0f);
        float childOriginY = originY + ((q & 2) ? childSize : 0.0f);
        float weight = std::max(0.0f, node.sum(q));
        ScreenRect rect = toScreenRect(childOriginX, childOriginY, childSize, canvasX, canvasY, canvasSize);

        if (node.isLeaf(q)) {
            float t = 0.0f;
            if (maxLeafWeight > 0.0f) {
                t = std::log1p(weight) / std::log1p(maxLeafWeight);
            }
            t = std::clamp(t, 0.0f, 1.0f);

            NVGcolor fill = nvgRGBAf(
                lerpFloat(0.06f, 0.96f, t),
                lerpFloat(0.10f, 0.58f, t),
                lerpFloat(0.16f, 0.10f, t),
                0.95f
            );
            NVGcolor stroke = nvgRGBAf(1.0f, 1.0f, 1.0f, 0.12f);
            drawRect(ctx, rect, fill, stroke, 1.0f);
        } else {
            float splitAlpha = std::clamp(0.05f + 0.02f * static_cast<float>(depthLevel), 0.05f, 0.22f);
            nvgBeginPath(ctx);
            nvgRect(ctx, rect.x, rect.y, rect.w, rect.h);
            nvgStrokeColor(ctx, nvgRGBAf(1.0f, 1.0f, 1.0f, splitAlpha));
            nvgStrokeWidth(ctx, 1.0f);
            nvgStroke(ctx);

            if (node.child(q) < numNodes) {
                drawTreeCell(ctx, nodes, numNodes, node.child(q), childOriginX, childOriginY, childSize,
                             canvasX, canvasY, canvasSize, maxLeafWeight, depthLevel + 1);
            }
        }
    }
}

} // namespace

class DTreeViewerScreen : public nanogui::Screen {
public:
    DTreeViewerScreen(int width, int height)
        : nanogui::Screen(nanogui::Vector2i(width, height), "Directional Guide Tree Visualizer") {
        setBackground(nanogui::Color(0.06f, 0.07f, 0.09f, 1.0f));

        m_panel = new nanogui::Window(this, "Tree Controls");
        m_panel->setPosition(nanogui::Vector2i(15, 15));
        m_panel->setFixedWidth(310);
        m_panel->setLayout(new nanogui::GroupLayout(12, 6, 6, 6));

        new nanogui::Label(m_panel, "Target Environment Lobe", "sans-bold");
        m_envCombo = new nanogui::ComboBox(m_panel, { "Single Sun Disk", "Dual Complex Lobes", "Structural Windows Cross" });
        m_envCombo->setCallback([this](int index) { m_env = static_cast<ELightingEnv>(index); });

        new nanogui::Label(m_panel, "Monte Carlo Sampling Type", "sans-bold");
        m_strategyCombo = new nanogui::ComboBox(m_panel, { "Uniform Exploration (1 / 4π)", "Guided Path Tracing (PPG)" });
        m_strategyCombo->setCallback([this](int index) { m_strategy = static_cast<ESamplingStrategy>(index); });

        new nanogui::Label(m_panel, "Visualization Canvas Mode", "sans-bold");
        m_viewCombo = new nanogui::ComboBox(m_panel, { "Sampling Tree (Active Guide)", "Building Tree (Accumulating)" });
        m_viewCombo->setCallback([this](int index) { m_viewMode = static_cast<EViewMode>(index); });

        new nanogui::Label(m_panel, "Max Subdivided Depth", "sans-bold");
        auto *depthBox = new nanogui::IntBox<int>(m_panel, m_maxDepth);
        depthBox->setMinMaxValues(1, 10);
        depthBox->setSpinnable(true);
        depthBox->setCallback([this](int val) { m_maxDepth = val; });

        new nanogui::Label(m_panel, "Refinement Threshold", "sans-bold");
        auto *thresholdBox = new nanogui::FloatBox<float>(m_panel, m_subdivisionThreshold);
        thresholdBox->setSpinnable(true);
        thresholdBox->setCallback([this](float val) { m_subdivisionThreshold = val; });

        new nanogui::Label(m_panel, "Simulation Actions", "sans-bold");
        
        auto *btn500 = new nanogui::Button(m_panel, "Trace & Record 500 Rays");
        btn500->setCallback([this] { recordRaysSimulation(500); });

        auto *btn5000 = new nanogui::Button(m_panel, "Trace & Record 5000 Rays");
        btn5000->setCallback([this] { recordRaysSimulation(5000); });

        auto *btnPass = new nanogui::Button(m_panel, "Commit Pass & Refine Topology");
        btnPass->setBackgroundColor(nanogui::Color(0.12f, 0.42f, 0.18f, 1.0f));
        btnPass->setCallback([this] { commitAndRefinePass(); });

        auto *btnReset = new nanogui::Button(m_panel, "Reset Whole Simulation");
        btnReset->setCallback([this] { resetSimulation(); });

        m_statsLabel = new nanogui::Label(m_panel, "", "sans", 13);
        m_statsLabel->setColor(nanogui::Color(0.8f, 0.85f, 0.9f, 1.0f));

        resetSimulation();
        performLayout();
    }

    void draw(NVGcontext *ctx) override {
        const float screenW = static_cast<float>(width());
        const float screenH = static_cast<float>(height());
        const float panelRight = static_cast<float>(m_panel->absolutePosition().x() + m_panel->width());
        const float margin = 30.0f;

        float canvasSize = std::min(screenW - panelRight - 2.0f * margin, screenH - 2.0f * margin);
        canvasSize = std::max(canvasSize, 200.0f);
        const float canvasX = panelRight + margin;
        const float canvasY = margin;

        nvgBeginPath(ctx);
        nvgRoundedRect(ctx, canvasX - 10.0f, canvasY - 10.0f, canvasSize + 20.0f, canvasSize + 20.0f, 12.0f);
        nvgFillColor(ctx, nvgRGBAf(0.03f, 0.04f, 0.06f, 1.0f));
        nvgFill(ctx);

        const DTree &activeTree = (m_viewMode == EViewMode::ESamplingTree) ? m_wrapper.sampling : m_wrapper.building;
        
        std::vector<QuadTreeNode> hostNodes(activeTree.numNodes());
        if (activeTree.numNodes() > 0 && activeTree.m_nodes != nullptr) {
            cudaMemcpy(hostNodes.data(), activeTree.m_nodes, hostNodes.size() * sizeof(QuadTreeNode), cudaMemcpyDeviceToHost);
        }

        TreeMetrics currentMetrics;
        if (!hostNodes.empty()) {
            collectMetrics(hostNodes.data(), hostNodes.size(), 0, 1, currentMetrics);
        }
        float maxWeight = std::max(1e-5f, currentMetrics.maxLeafWeight);

        if (!hostNodes.empty()) {
            drawTreeCell(ctx, hostNodes.data(), hostNodes.size(), 0, 0.0f, 0.0f, 1.0f, canvasX, canvasY, canvasSize, maxWeight, 1);
        }

        if (!m_visualPoints.empty()) {
            for (const auto &p : m_visualPoints) {
                float px = canvasX + p.x * canvasSize;
                float py = canvasY + (1.0f - p.y) * canvasSize;
                nvgBeginPath(ctx);
                nvgCircle(ctx, px, py, 1.85f);
                nvgFillColor(ctx, nvgRGBAf(1.0f, 1.0f, 1.0f, 0.55f));
                nvgFill(ctx);
            }
        }

        nvgBeginPath(ctx);
        nvgRoundedRect(ctx, canvasX, canvasY, canvasSize, canvasSize, 2.0f);
        nvgStrokeColor(ctx, nvgRGBAf(0.8f, 0.85f, 0.9f, 0.25f));
        nvgStrokeWidth(ctx, 1.5f);
        nvgStroke(ctx);

        nvgFontFace(ctx, "sans-bold");
        nvgFontSize(ctx, 18.0f);
        nvgFillColor(ctx, nvgRGBAf(0.95f, 0.95f, 0.95f, 1.0f));
        nvgText(ctx, canvasX, canvasY - 14.0f, 
                m_viewMode == EViewMode::ESamplingTree ? "Sampling Tree (Active Guide)" : "Building Tree (Accumulated Estimates Pass)", 
                nullptr);

        Screen::draw(ctx);
    }

private:
    void resetSimulation() {
        m_wrapper.building.clear();
        m_wrapper.building.initialize();
        m_wrapper.sampling.clear();
        m_wrapper.sampling.initialize();
        m_wrapper.build(); 

        m_totalRaysRecorded = 0;
        m_visualPoints.clear();
        updateStatusText(0, 0);
    }

    void recordRaysSimulation(int count) {
        futaba::Sampler sampler(m_seedCounter++);
        m_visualPoints.clear();

        std::vector<QuadTreeNode> hostSampling(m_wrapper.sampling.numNodes());
        if (m_wrapper.sampling.numNodes() > 0 && m_wrapper.sampling.m_nodes != nullptr) {
            cudaMemcpy(hostSampling.data(), m_wrapper.sampling.m_nodes, hostSampling.size() * sizeof(QuadTreeNode), cudaMemcpyDeviceToHost);
        }

        std::vector<QuadTreeNode> hostBuilding(m_wrapper.building.numNodes());
        if (m_wrapper.building.numNodes() > 0 && m_wrapper.building.m_nodes != nullptr) {
            cudaMemcpy(hostBuilding.data(), m_wrapper.building.m_nodes, hostBuilding.size() * sizeof(QuadTreeNode), cudaMemcpyDeviceToHost);
        }

        int hitCount = 0;
        for (int i = 0; i < count; ++i) {
            Vector3f dir;
            float woPdf = 1.0f;
            Point2f canonicalPoint;

            if (m_strategy == ESamplingStrategy::EUniform) {
                canonicalPoint = sampler.next2D();
                dir = DTreeWrapper::canonicalToDir(canonicalPoint);
                woPdf = 1.0f / (4.0f * M_PI);
            } else {
                if (m_wrapper.sampling.mean() > 0.0f && !hostSampling.empty()) {
                    canonicalPoint = hostSampling[0].sample(sampler, hostSampling.data());
                    canonicalPoint.x = std::clamp(canonicalPoint.x, 0.0f, 1.0f);
                    canonicalPoint.y = std::clamp(canonicalPoint.y, 0.0f, 1.0f);
                } else {
                    canonicalPoint = sampler.next2D();
                }

                dir = DTreeWrapper::canonicalToDir(canonicalPoint);

                if (m_wrapper.sampling.mean() > 0.0f && !hostSampling.empty()) {
                    woPdf = (hostSampling[0].pdf(DTreeWrapper::dirToCanonical(dir), hostSampling.data())) / (4.0f * M_PI);
                } else {
                    woPdf = 1.0f / (4.0f * M_PI);
                }

                if (woPdf <= 0.0f) {
                    woPdf = 1.0f / (4.0f * M_PI);
                }
            }

            float radiance = evaluateTargetRadiance(dir, m_env);
            float statisticalWeight = 1.0f;

            if (radiance > 0.0f && woPdf > 0.0f && !hostBuilding.empty()) {
                float irradiance = radiance / woPdf;
                Point2f pRecord = DTreeWrapper::dirToCanonical(dir);

                m_wrapper.building.m_statisticalWeight += statisticalWeight;
                m_wrapper.building.m_sum += irradiance * statisticalWeight;

                hostBuilding[0].record(pRecord, irradiance * statisticalWeight, hostBuilding.data());
            }

            m_visualPoints.push_back(DTreeWrapper::dirToCanonical(dir));
            if (radiance > 0.5f) hitCount++;
        }

        if (!hostBuilding.empty()) {
            cudaMemcpy(m_wrapper.building.m_nodes, hostBuilding.data(), hostBuilding.size() * sizeof(QuadTreeNode), cudaMemcpyHostToDevice);
        }

        m_totalRaysRecorded += count;
        updateStatusText(count, hitCount);
    }

    void commitAndRefinePass() {
        m_wrapper.build();
        m_wrapper.reset(m_maxDepth, m_subdivisionThreshold);
        
        m_visualPoints.clear();
        updateStatusText(0, 0);
    }

    void updateStatusText(int lastBatch, int hitCount) {
        std::vector<QuadTreeNode> sHostNodes(m_wrapper.sampling.numNodes());
        if (m_wrapper.sampling.numNodes() > 0 && m_wrapper.sampling.m_nodes != nullptr) {
            cudaMemcpy(sHostNodes.data(), m_wrapper.sampling.m_nodes, sHostNodes.size() * sizeof(QuadTreeNode), cudaMemcpyDeviceToHost);
        }

        std::vector<QuadTreeNode> bHostNodes(m_wrapper.building.numNodes());
        if (m_wrapper.building.numNodes() > 0 && m_wrapper.building.m_nodes != nullptr) {
            cudaMemcpy(bHostNodes.data(), m_wrapper.building.m_nodes, bHostNodes.size() * sizeof(QuadTreeNode), cudaMemcpyDeviceToHost);
        }

        TreeMetrics sMetrics, bMetrics;
        if (!sHostNodes.empty()) collectMetrics(sHostNodes.data(), sHostNodes.size(), 0, 1, sMetrics);
        if (!bHostNodes.empty()) collectMetrics(bHostNodes.data(), bHostNodes.size(), 0, 1, bMetrics);

        std::string text = 
            "Total Sampled Rays: " + std::to_string(m_totalRaysRecorded) + "\n" +
            "Last Batch Hits: " + std::to_string(hitCount) + " / " + std::to_string(lastBatch) + "\n\n" +
            "--- SAMPLING TREE (Active Guide) ---\n" +
            "Nodes: " + std::to_string(m_wrapper.sampling.numNodes()) + " | Leaves: " + std::to_string(sMetrics.leafCount) + "\n" +
            "Max Depth: " + std::to_string(m_wrapper.sampling.depth()) + "\n" +
            "Total Accumulated Energy: " + std::to_string(sMetrics.totalWeight) + "\n\n" +
            "--- BUILDING TREE (Next Pass) ---\n" +
            "Nodes: " + std::to_string(m_wrapper.building.numNodes()) + " | Leaves: " + std::to_string(bMetrics.leafCount) + "\n" +
            "Max Depth: " + std::to_string(m_wrapper.building.depth());

        m_statsLabel->setCaption(text);
        m_panel->performLayout(screen()->nvgContext());
    }

private:
    nanogui::Window *m_panel = nullptr;
    nanogui::Label *m_statsLabel = nullptr;
    nanogui::ComboBox *m_envCombo = nullptr;
    nanogui::ComboBox *m_strategyCombo = nullptr;
    nanogui::ComboBox *m_viewCombo = nullptr;

    DTreeWrapper m_wrapper;
    std::vector<Point2f> m_visualPoints;
    
    ELightingEnv m_env = ELightingEnv::ESingleSpot;
    ESamplingStrategy m_strategy = ESamplingStrategy::EUniform;
    EViewMode m_viewMode = EViewMode::ESamplingTree;

    int m_maxDepth = 6;
    float m_subdivisionThreshold = 0.01f;
    int m_totalRaysRecorded = 0;
    unsigned m_seedCounter = 1337;
};

int main() {
    try {
        nanogui::init();
        {
            nanogui::ref<DTreeViewerScreen> screen = new DTreeViewerScreen(1280, 760);
            screen->drawAll();
            screen->setVisible(true);
            nanogui::mainloop();
        }
        nanogui::shutdown();
    } catch (const std::exception &e) {
        std::cerr << "Fatal deployment runtime error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}