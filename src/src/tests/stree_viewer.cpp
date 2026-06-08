#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nanogui/opengl.h>
#include <nanogui/glcanvas.h>
#include <nanogui/glutil.h>
#include <nanogui/screen.h>
#include <nanogui/window.h>
#include <nanogui/layout.h>
#include <nanogui/label.h>
#include <nanogui/button.h>
#include <nanogui/checkbox.h>
#include <nanogui/textbox.h>

#include "ppg/stree.h"

// Explicitly register types matching the guiding architecture definitions
using namespace futaba;

namespace {

struct TreeMetrics {
    size_t leafCount = 0;
    size_t pointCount = 0;
    size_t nodeCount = 0;
    float totalWeight = 0.0f;
    float maxLeafWeight = 0.0f;
    float maxSampleWeight = 0.0f;
};

struct LeafVisual {
    Point3f center = Point3f(0.0f, 0.0f, 0.0f);
    Vector3f size = Vector3f(0.0f, 0.0f, 0.0f);
    float weight = 0.0f;
    float meanRadiance = 0.0f;
};

struct SampleRecord {
    Point3f position = Point3f(0.0f, 0.0f, 0.0f);
    DTreeRecord record;
};

struct SceneGeometry {
    nanogui::MatrixXf linePositions;
    nanogui::MatrixXf lineColors;
    nanogui::MatrixXf pointPositions;
    nanogui::MatrixXf pointColors;
};

static futaba::STree makeUnitTree() {
    futaba::AABB bounds;
    bounds.minP = Point3f(0.0f, 0.0f, 0.0f);
    bounds.maxP = Point3f(1.0f, 1.0f, 1.0f);
    return futaba::STree(bounds);
}

static Point3f clampPoint(const Point3f &p) {
    return Point3f(
        std::clamp(p.x, 0.0f, 1.0f),
        std::clamp(p.y, 0.0f, 1.0f),
        std::clamp(p.z, 0.0f, 1.0f));
}

static Vector3f randomUnitVector(std::mt19937 &rng) {
    std::uniform_real_distribution<float> unit01(0.0f, 1.0f);
    const float z = unit01(rng) * 2.0f - 1.0f;
    const float phi = unit01(rng) * 2.0f * static_cast<float>(M_PI);
    const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    return Vector3f(r * std::cos(phi), r * std::sin(phi), z);
}

static nanogui::Color weightColor(float t, float alpha) {
    t = std::clamp(t, 0.0f, 1.0f);
    return nanogui::Color(
        0.18f + 0.76f * t,
        0.26f + 0.44f * t,
        0.40f + 0.12f * (1.0f - t),
        alpha);
}

static SampleRecord makeSampleRecord(unsigned seed, size_t sampleIndex) {
    std::mt19937 rng(seed ^ static_cast<unsigned>(sampleIndex * 0x9e3779b9u + 0x85ebca6bu));
    std::normal_distribution<float> jitter(0.0f, 1.0f);
    std::uniform_real_distribution<float> unit01(0.0f, 1.0f);

    const std::array<Point3f, 3> clusters = {
        Point3f(0.22f, 0.24f, 0.28f),
        Point3f(0.72f, 0.34f, 0.58f),
        Point3f(0.44f, 0.78f, 0.36f)
    };

    const float pointSpread = 0.085f;
    const int clusterIndex = static_cast<int>(sampleIndex % clusters.size());
    const Point3f center = clusters[static_cast<size_t>(clusterIndex)];

    SampleRecord sample;
    sample.position = Point3f(
        std::clamp(center.x + jitter(rng) * pointSpread, 0.0f, 1.0f),
        std::clamp(center.y + jitter(rng) * pointSpread, 0.0f, 1.0f),
        std::clamp(center.z + jitter(rng) * pointSpread, 0.0f, 1.0f));

    const float clusterFactor = 0.55f + 0.45f * unit01(rng);
    sample.record.d = Vector3f(0.0f, 0.0f, 1.0f);
    sample.record.radiance = 0.18f + 1.7f * clusterFactor;
    sample.record.product = 1.0f;
    sample.record.woPdf = 1.0f;
    sample.record.bsdfPdf = 1.0f;
    sample.record.dTreePdf = 1.0f;
    sample.record.statisticalWeight = 0.6f + 1.1f * clusterFactor;
    sample.record.isDelta = false;

    return sample;
}

static void addCubeWireframe(const Point3f &center,
                             const Vector3f &size,
                             const nanogui::Color &color,
                             nanogui::MatrixXf &linePositions,
                             nanogui::MatrixXf &lineColors,
                             size_t &column) {
    const Vector3f half = 0.5f * size;
    const Point3f minP = center - half;
    const Point3f maxP = center + half;

    std::array<Point3f, 8> corners = {
        Point3f(minP.x, minP.y, minP.z),
        Point3f(maxP.x, minP.y, minP.z),
        Point3f(maxP.x, maxP.y, minP.z),
        Point3f(minP.x, maxP.y, minP.z),
        Point3f(minP.x, minP.y, maxP.z),
        Point3f(maxP.x, minP.y, maxP.z),
        Point3f(maxP.x, maxP.y, maxP.z),
        Point3f(minP.x, maxP.y, maxP.z)
    };

    static const std::array<std::pair<int, int>, 12> edges = {{
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7}
    }};

    for (const auto &edge : edges) {
        const Point3f &a = corners[edge.first];
        const Point3f &b = corners[edge.second];

        linePositions.col(column) << a.x, a.y, a.z;
        lineColors.col(column) << color.r(), color.g(), color.b(), color.w();
        ++column;

        linePositions.col(column) << b.x, b.y, b.z;
        lineColors.col(column) << color.r(), color.g(), color.b(), color.w();
        ++column;
    }
}

static void initShader(nanogui::GLShader &shader, const std::string &name) {
    shader.init(
        name,
        "#version 330\n"
        "uniform mat4 modelViewProj;\n"
        "uniform float pointSize;\n"
        "in vec3 position;\n"
        "in vec4 color;\n"
        "out vec4 frag_color;\n"
        "void main() {\n"
        "    frag_color = color;\n"
        "    gl_Position = modelViewProj * vec4(position, 1.0);\n"
        "    gl_PointSize = pointSize;\n"
        "}\n",
        "#version 330\n"
        "in vec4 frag_color;\n"
        "out vec4 out_color;\n"
        "void main() {\n"
        "    out_color = frag_color;\n"
        "}\n");
}

static void hostRecordOnSTree(futaba::STree& tree, const Point3f& p, const DTreeRecord& rec, EDirectionalFilter dirFilter) {
    std::vector<futaba::STreeNode> hostSTreeNodes(tree.numNodes());
    if (tree.numNodes() > 0 && tree.m_nodes != nullptr) {
        cudaMemcpy(hostSTreeNodes.data(), tree.m_nodes, hostSTreeNodes.size() * sizeof(futaba::STreeNode), cudaMemcpyDeviceToHost);
    }

    Vector3f size = Vector3f(tree.m_aabb.maxP.x - tree.m_aabb.minP.x,
                             tree.m_aabb.maxP.y - tree.m_aabb.minP.y,
                             tree.m_aabb.maxP.z - tree.m_aabb.minP.z);
    Point3f pLocal = p;
    pLocal.x = (pLocal.x - tree.m_aabb.minP.x) / size.x;
    pLocal.y = (pLocal.y - tree.m_aabb.minP.y) / size.y;
    pLocal.z = (pLocal.z - tree.m_aabb.minP.z) / size.z;

    size_t currentIdx = 0;
    while (!hostSTreeNodes[currentIdx].isLeaf) {
        currentIdx = hostSTreeNodes[currentIdx].nodeIndex(pLocal);
    }

    futaba::STreeNode& targetLeaf = hostSTreeNodes[currentIdx];
    if (rec.isDelta) return;
    if (!(rec.woPdf > 0.0f)) return;

    float irradiance = rec.radiance / rec.woPdf;
    Point2f pDirectional = DTreeWrapper::dirToCanonical(rec.d);
    float statisticalWeight = rec.statisticalWeight;

    targetLeaf.dTree.building.m_statisticalWeight += statisticalWeight;
    targetLeaf.dTree.building.m_sum += irradiance * statisticalWeight;

    size_t numQuadNodes = targetLeaf.dTree.building.numNodes();
    std::vector<QuadTreeNode> hostQuadNodes(numQuadNodes);
    if (numQuadNodes > 0 && targetLeaf.dTree.building.m_nodes != nullptr) {
        cudaMemcpy(hostQuadNodes.data(), targetLeaf.dTree.building.m_nodes, numQuadNodes * sizeof(QuadTreeNode), cudaMemcpyDeviceToHost);
    }

    if (!hostQuadNodes.empty()) {
        if (dirFilter == EDirectionalFilter::ENearest) {
            hostQuadNodes[0].record(pDirectional, irradiance * statisticalWeight, hostQuadNodes.data());
        } else {
            int currentDepth = hostQuadNodes[0].depthAt(pDirectional, hostQuadNodes.data());
            float sizeQuad = std::pow(0.5f, currentDepth);
            Point2f originQuad = pDirectional;
            originQuad.x -= sizeQuad / 2.0f;
            originQuad.y -= sizeQuad / 2.0f;
            hostQuadNodes[0].record(originQuad, sizeQuad, Point2f(0.0f), 1.0f, irradiance * statisticalWeight / (sizeQuad * sizeQuad), hostQuadNodes.data());
        }
        cudaMemcpy(targetLeaf.dTree.building.m_nodes, hostQuadNodes.data(), numQuadNodes * sizeof(QuadTreeNode), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(tree.m_nodes, hostSTreeNodes.data(), hostSTreeNodes.size() * sizeof(futaba::STreeNode), cudaMemcpyHostToDevice);
}

class STreeCanvas : public nanogui::GLCanvas {
public:
    explicit STreeCanvas(Widget *parent)
        : nanogui::GLCanvas(parent) {
        initShader(m_lineShader, "stree_lines");
        initShader(m_pointShader, "stree_points");
        setDrawBorder(true);
        setBackgroundColor(nanogui::Color(0.04f, 0.05f, 0.07f, 1.0f));
    }

    ~STreeCanvas() override {
        m_lineShader.free();
        m_pointShader.free();
    }

    void resetCamera() {
        m_arcball.setState(Eigen::Quaternionf::Identity());
        m_distance = 2.8f;
    }

    void setShowPoints(bool value) {
        m_showPoints = value;
    }

    void setGeometry(const SceneGeometry &geometry) {
        m_lineVertexCount = static_cast<size_t>(geometry.linePositions.cols());
        m_pointVertexCount = static_cast<size_t>(geometry.pointPositions.cols());

        m_lineShader.bind();
        m_lineShader.uploadAttrib("position", geometry.linePositions);
        m_lineShader.uploadAttrib("color", geometry.lineColors);

        m_pointShader.bind();
        m_pointShader.uploadAttrib("position", geometry.pointPositions);
        m_pointShader.uploadAttrib("color", geometry.pointColors);
    }

    bool mouseButtonEvent(const nanogui::Vector2i &p, int button, bool down, int modifiers) override {
        if (button == GLFW_MOUSE_BUTTON_1) {
            const nanogui::Vector2i localPos = p - absolutePosition();
            m_arcball.button(localPos, down);
            return true;
        }
        return Widget::mouseButtonEvent(p, button, down, modifiers);
    }

    bool mouseMotionEvent(const nanogui::Vector2i &p, const nanogui::Vector2i &rel, int button, int modifiers) override {
        if (button & (1 << GLFW_MOUSE_BUTTON_1)) {
            const nanogui::Vector2i localPos = p - absolutePosition();
            return m_arcball.motion(localPos);
        }
        return Widget::mouseMotionEvent(p, rel, button, modifiers);
    }

    bool mouseDragEvent(const nanogui::Vector2i &p, const nanogui::Vector2i &rel, int button, int modifiers) override {
        if (button & (1 << GLFW_MOUSE_BUTTON_1)) {
            const nanogui::Vector2i localPos = p - absolutePosition();
            return m_arcball.motion(localPos);
        }
        return Widget::mouseDragEvent(p, rel, button, modifiers);
    }

    bool scrollEvent(const nanogui::Vector2i &p, const nanogui::Vector2f &rel) override {
        m_distance = std::clamp(m_distance - rel.y() * 0.18f, 1.4f, 8.0f);
        return true;
    }

    void drawGL() override {
        const nanogui::Vector2i canvasSize = size();
        if (canvasSize.x() <= 0 || canvasSize.y() <= 0) {
            return;
        }

        m_arcball.setSize(canvasSize);

        const float aspect = static_cast<float>(canvasSize.x()) / static_cast<float>(canvasSize.y());
        const float fovY = 45.0f * static_cast<float>(M_PI) / 180.0f;
        const float nearPlane = 0.05f;
        const float farPlane = 20.0f;

        nanogui::Matrix4f projection = nanogui::Matrix4f::Zero();
        const float f = 1.0f / std::tan(fovY * 0.5f);
        projection(0, 0) = f / aspect;
        projection(1, 1) = f;
        projection(2, 2) = (farPlane + nearPlane) / (nearPlane - farPlane);
        projection(2, 3) = (2.0f * farPlane * nearPlane) / (nearPlane - farPlane);
        projection(3, 2) = -1.0f;

        nanogui::Matrix4f model = nanogui::Matrix4f::Identity();
        model(0, 3) = -0.5f;
        model(1, 3) = -0.5f;
        model(2, 3) = -0.5f;

        nanogui::Matrix4f view = nanogui::Matrix4f::Identity();
        view(2, 3) = -m_distance;
        view = view * m_arcball.matrix();

        const nanogui::Matrix4f mvp = projection * view * model;

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_PROGRAM_POINT_SIZE);

        if (m_lineVertexCount > 0) {
            m_lineShader.bind();
            m_lineShader.setUniform("modelViewProj", mvp);
            m_lineShader.setUniform("pointSize", 1.0f);
            m_lineShader.drawArray(GL_LINES, 0, static_cast<uint32_t>(m_lineVertexCount));
        }

        if (m_showPoints && m_pointVertexCount > 0) {
            m_pointShader.bind();
            m_pointShader.setUniform("modelViewProj", mvp);
            m_pointShader.setUniform("pointSize", 6.0f);
            m_pointShader.drawArray(GL_POINTS, 0, static_cast<uint32_t>(m_pointVertexCount));
        }

        glDisable(GL_PROGRAM_POINT_SIZE);
        glDisable(GL_BLEND);
        glDisable(GL_DEPTH_TEST);
    }

private:
    nanogui::GLShader m_lineShader;
    nanogui::GLShader m_pointShader;
    nanogui::Arcball m_arcball;
    size_t m_lineVertexCount = 0;
    size_t m_pointVertexCount = 0;
    float m_distance = 2.8f;
    bool m_showPoints = true;
};

static void buildSceneGeometry(const futaba::STree &tree,
                               const std::vector<Point3f> &samplePoints,
                               const std::vector<float> &sampleWeights,
                               SceneGeometry &geometry,
                               TreeMetrics &metrics) {
    std::vector<LeafVisual> leaves;
    leaves.reserve(tree.numLeaves());

    tree.forEachDTreeWrapperConstP(
        [&](const DTreeWrapper *dw, const Point3f &minCorner, const Vector3f &size) {
            LeafVisual leaf;
            leaf.center = minCorner + 0.5f * size;
            leaf.size = size;
            leaf.weight = dw->statisticalWeight();
            leaf.meanRadiance = dw->meanRadiance();
            leaves.push_back(leaf);

            metrics.leafCount += 1;
            metrics.totalWeight += leaf.weight;
            metrics.maxLeafWeight = std::max(metrics.maxLeafWeight, leaf.weight);
        });

    metrics.nodeCount = tree.numNodes();
    metrics.pointCount = samplePoints.size();

    geometry.linePositions.resize(3, leaves.size() * 24);
    geometry.lineColors.resize(4, leaves.size() * 24);
    geometry.pointPositions.resize(3, samplePoints.size());
    geometry.pointColors.resize(4, samplePoints.size());

    size_t lineColumn = 0;
    const float maxWeight = std::max(1e-6f, metrics.maxLeafWeight);
    for (const auto &leaf : leaves) {
        const float t = std::log1p(std::max(0.0f, leaf.weight)) / std::log1p(maxWeight);
        const nanogui::Color lineColor = weightColor(t, 0.26f + 0.52f * t);
        addCubeWireframe(leaf.center, leaf.size, lineColor,
                         geometry.linePositions, geometry.lineColors, lineColumn);
    }

    for (size_t i = 0; i < samplePoints.size(); ++i) {
        const Point3f &p = samplePoints[i];
        const float weight = (i < sampleWeights.size()) ? sampleWeights[i] : 1.0f;
        const float t = std::log1p(std::max(0.0f, weight)) / std::log1p(std::max(1.0f, metrics.maxLeafWeight));
        const nanogui::Color pointColor = weightColor(t, 0.92f);

        geometry.pointPositions.col(static_cast<Eigen::Index>(i)) << p.x, p.y, p.z;
        geometry.pointColors.col(static_cast<Eigen::Index>(i)) << pointColor.r(), pointColor.g(), pointColor.b(), pointColor.w();
    }
}

static std::string formatStatus(const TreeMetrics &metrics,
                                int seed,
                                int sampleCount,
                                int refinePasses,
                                int splitThreshold,
                                bool showPoints) {
    std::ostringstream out;
    out << "Seed: " << seed
        << "   Samples: " << sampleCount
        << "   Refinements: " << refinePasses
        << "   Threshold: " << splitThreshold
        << "   Leaves: " << metrics.leafCount
        << "   Nodes: " << metrics.nodeCount
        << "   Points: " << metrics.pointCount
        << "   Total weight: " << metrics.totalWeight
        << "   Max leaf weight: " << metrics.maxLeafWeight
        << "   Points visible: " << (showPoints ? "yes" : "no");
    return out.str();
}

} // namespace

class STreeViewerScreen : public nanogui::Screen {
public:
    STreeViewerScreen(int width, int height)
        : nanogui::Screen(nanogui::Vector2i(width, height), "S-Tree Viewer")
        , m_tree(makeUnitTree()) {
        setBackground(nanogui::Color(0.06f, 0.07f, 0.09f, 1.0f));

        m_panel = new nanogui::Window(this, "Tree Controls");
        m_panel->setPosition(nanogui::Vector2i(15, 15));
        m_panel->setFixedWidth(320);
        m_panel->setLayout(new nanogui::GroupLayout(10, 5, 5, 5));

        new nanogui::Label(m_panel, "Accumulated spatial path-guiding tree", "sans-bold");
        m_statusLabel = new nanogui::Label(m_panel, "");

        new nanogui::Label(m_panel, "Seed", "sans-bold");
        m_seedBox = new nanogui::IntBox<int>(m_panel, m_seed);
        m_seedBox->setFixedWidth(120);
        m_seedBox->setSpinnable(true);
        m_seedBox->setCallback([this](int value) {
            m_seed = value;
            updateStatusLabel();
        });

        new nanogui::Label(m_panel, "Add count", "sans-bold");
        m_batchBox = new nanogui::IntBox<int>(m_panel, m_batchCount);
        m_batchBox->setFixedWidth(120);
        m_batchBox->setMinMaxValues(1, 5000);
        m_batchBox->setSpinnable(true);
        m_batchBox->setCallback([this](int value) {
            m_batchCount = std::max(1, value);
            updateStatusLabel();
        });

        new nanogui::Label(m_panel, "Refinement passes", "sans-bold");
        m_refineBox = new nanogui::IntBox<int>(m_panel, m_refinePasses);
        m_refineBox->setFixedWidth(120);
        m_refineBox->setMinMaxValues(0, 8);
        m_refineBox->setSpinnable(true);
        m_refineBox->setCallback([this](int value) {
            m_refinePasses = value;
            rebuildScene();
        });

        new nanogui::Label(m_panel, "Split threshold", "sans-bold");
        m_thresholdBox = new nanogui::IntBox<int>(m_panel, m_splitThreshold);
        m_thresholdBox->setFixedWidth(120);
        m_thresholdBox->setMinMaxValues(1, 1000);
        m_thresholdBox->setSpinnable(true);
        m_thresholdBox->setCallback([this](int value) {
            m_splitThreshold = std::max(1, value);
            rebuildScene();
        });

        auto *addButton = new nanogui::Button(m_panel, "Add samples");
        addButton->setCallback([this] { appendSamples(); });

        auto *resetButton = new nanogui::Button(m_panel, "Reset samples");
        resetButton->setCallback([this] { resetSamples(); });

        m_showPointsCheck = new nanogui::CheckBox(m_panel, "Show points");
        m_showPointsCheck->setChecked(true);
        m_showPointsCheck->setCallback([this](bool value) {
            if (m_canvas) {
                m_canvas->setShowPoints(value);
            }
        });

        auto *rebuildButton = new nanogui::Button(m_panel, "Rebuild scene");
        rebuildButton->setCallback([this] { rebuildScene(); });

        auto *cameraButton = new nanogui::Button(m_panel, "Reset camera");
        cameraButton->setCallback([this] {
            if (m_canvas) {
                m_canvas->resetCamera();
            }
        });

        new nanogui::Label(m_panel,
            "Drag with the left mouse button to orbit. Scroll to zoom."
        );

        m_canvas = new STreeCanvas(this);
        m_canvas->setPosition(nanogui::Vector2i(0, 0));
        m_canvas->setSize(nanogui::Vector2i(1, 1));

        rebuildScene();
        performLayout();
    }

    void draw(NVGcontext *ctx) override {
        layoutCanvas();
        Screen::draw(ctx);
    }

private:
    void layoutCanvas() {
        if (!m_canvas || !m_panel) {
            return;
        }

        const float margin = 26.0f;
        const float panelRight = static_cast<float>(m_panel->absolutePosition().x() + m_panel->width());
        const float canvasX = panelRight + margin;
        const float canvasY = margin;
        const float canvasW = std::max(260.0f, static_cast<float>(width()) - canvasX - margin);
        const float canvasH = std::max(260.0f, static_cast<float>(height()) - 2.0f * margin);

        m_canvas->setPosition(nanogui::Vector2i(static_cast<int>(canvasX), static_cast<int>(canvasY)));
        m_canvas->setSize(nanogui::Vector2i(static_cast<int>(canvasW), static_cast<int>(canvasH)));
    }

    void appendSamples() {
        const size_t startIndex = m_records.size();
        m_records.reserve(startIndex + static_cast<size_t>(m_batchCount));

        for (int i = 0; i < m_batchCount; ++i) {
            m_records.push_back(makeSampleRecord(static_cast<unsigned>(m_seed), startIndex + static_cast<size_t>(i)));
        }

        rebuildScene();
    }

    void resetSamples() {
        m_records.clear();
        rebuildScene();
    }

    void rebuildScene() {
        m_tree.clear();
        m_tree = makeUnitTree();
        m_metrics = {};
        m_samplePoints.clear();
        m_sampleWeights.clear();

        std::vector<SampleRecord> sampleRecords = m_records;
        for (const SampleRecord &sample : sampleRecords) {
            m_samplePoints.push_back(sample.position);
            m_sampleWeights.push_back(sample.record.statisticalWeight);
            m_metrics.maxSampleWeight = std::max(m_metrics.maxSampleWeight, sample.record.statisticalWeight);
        }

        if (!sampleRecords.empty()) {
            const int batchCount = std::max(1, m_refinePasses + 1);
            const int totalSamples = static_cast<int>(sampleRecords.size());
            int sampleBegin = 0;

            for (int pass = 0; pass < batchCount; ++pass) {
                const int sampleEnd = (pass + 1 == batchCount)
                    ? totalSamples
                    : (totalSamples * (pass + 1) / batchCount);

                for (int i = sampleBegin; i < sampleEnd; ++i) {
                    hostRecordOnSTree(m_tree, m_samplePoints[static_cast<size_t>(i)],
                                      sampleRecords[static_cast<size_t>(i)].record,
                                      EDirectionalFilter::ENearest);
                }

                sampleBegin = sampleEnd;

                if (pass + 1 < batchCount) {
                    m_tree.refine(static_cast<size_t>(m_splitThreshold), -1);
                }
            }
        }

        SceneGeometry geometry;
        buildSceneGeometry(m_tree, m_samplePoints, m_sampleWeights, geometry, m_metrics);

        if (m_canvas) {
            m_canvas->setGeometry(geometry);
            m_canvas->setShowPoints(m_showPointsCheck ? m_showPointsCheck->checked() : true);
            m_canvas->resetCamera();
        }

        updateStatusLabel();
        performLayout();
    }

    void updateStatusLabel() {
        if (m_statusLabel) {
            m_statusLabel->setCaption(formatStatus(
                m_metrics,
                m_seed,
                static_cast<int>(m_records.size()),
                m_refinePasses,
                m_splitThreshold,
                m_showPointsCheck ? m_showPointsCheck->checked() : true));
        }
    }

private:
    nanogui::Window *m_panel = nullptr;
    nanogui::Label *m_statusLabel = nullptr;
    nanogui::IntBox<int> *m_seedBox = nullptr;
    nanogui::IntBox<int> *m_batchBox = nullptr;
    nanogui::IntBox<int> *m_refineBox = nullptr;
    nanogui::IntBox<int> *m_thresholdBox = nullptr;
    nanogui::CheckBox *m_showPointsCheck = nullptr;
    STreeCanvas *m_canvas = nullptr;

    futaba::STree m_tree;
    TreeMetrics m_metrics;
    std::vector<SampleRecord> m_records;
    std::vector<Point3f> m_samplePoints;
    std::vector<float> m_sampleWeights;

    int m_seed = 1337;
    int m_batchCount = 1;
    int m_refinePasses = 4;
    int m_splitThreshold = 48;
};

int main() {
    try {
        nanogui::init();

        {
            nanogui::ref<STreeViewerScreen> screen = new STreeViewerScreen(1280, 800);
            screen->drawAll();
            screen->setVisible(true);
            nanogui::mainloop();
        }

        nanogui::shutdown();
    } catch (const std::exception &e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}