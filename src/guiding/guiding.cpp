#include "guiding.h"
#include "launch_params.h"
#include "common.cuh"

#include <nanogui/nanogui.h>

#include <algorithm>
#include <cmath>
#include <vector>

FUTABA_NAMESPACE_BEGIN

// ---------------------------------------------------------------------------
// None: guiding disabled.
// ---------------------------------------------------------------------------
class NoneGuiding : public GuidingMethod {
public:
    std::string     name() const override { return "None"; }
    PathGuidingMode mode() const override { return PATH_GUIDING_NONE; }

    void updateLaunchParams(LaunchParams& params) const override {
        params.guiding.active     = false;
        params.guiding.type       = GuidingType::None;
        params.guiding.sTreeNodes = nullptr;
    }
};

// ---------------------------------------------------------------------------
// PPG: Practical Path Guiding (SD-tree). Built on the CPU from the captured
// training samples, sampled on the GPU. Minimal variant -- radiance target,
// nearest deposition, no directional/spatial filtering.
// ---------------------------------------------------------------------------
class PPGGuiding : public GuidingMethod {
public:
    ~PPGGuiding() override { delete m_sTree; }

    std::string     name() const override { return "PPG"; }
    PathGuidingMode mode() const override { return PATH_GUIDING_PPG; }

    void setSceneBounds(const AABB& bounds) override {
        m_bounds = valid(bounds) ? bounds : defaultBounds();
        rebuild();
    }

    void reset() override { rebuild(); }

    void train(const TrainingBufferManager& buf, int maxDepth) override {
        if (!m_sTree || maxDepth <= 0 || buf.count() == 0) return;
        const size_t count = buf.count();
        const size_t numS  = m_sTree->m_numNodes;
        if (numS == 0) return;

        // Read back captured samples from device
        std::vector<float>    active(count);
        std::vector<Point3f>  position(count);
        std::vector<Vector3f> wo(count);
        std::vector<Color3f>  radiance(count);
        std::vector<float>    pdf(count);
        
        cudaMemcpy(active.data(),   buf.active(),   count * sizeof(float),    cudaMemcpyDeviceToHost);
        cudaMemcpy(position.data(), buf.position(), count * sizeof(Point3f),  cudaMemcpyDeviceToHost);
        cudaMemcpy(wo.data(),       buf.wo(),       count * sizeof(Vector3f), cudaMemcpyDeviceToHost);
        cudaMemcpy(radiance.data(), buf.radiance(), count * sizeof(Color3f),  cudaMemcpyDeviceToHost);
        cudaMemcpy(pdf.data(),      buf.pdf(),      count * sizeof(float),    cudaMemcpyDeviceToHost);

        std::vector<STreeNode> sNodes(numS);
        cudaMemcpy(sNodes.data(), m_sTree->m_nodes, numS * sizeof(STreeNode), cudaMemcpyDeviceToHost);

        // Host copy of each leaf's building quadtree nodes.
        std::vector<std::vector<QuadTreeNode>> leafNodes(numS);
        for (size_t s = 0; s < numS; ++s) {
            if (!sNodes[s].isLeaf) continue;
            DTree& b = sNodes[s].dTree.building;
            leafNodes[s].resize(b.m_numNodes > 0 ? b.m_numNodes : 1);
            if (b.m_numNodes > 0 && b.m_nodes)
                cudaMemcpy(leafNodes[s].data(), b.m_nodes, b.m_numNodes * sizeof(QuadTreeNode), cudaMemcpyDeviceToHost);
        }

        // Splat captured samples into building trees
        for (size_t i = 0; i < count; ++i) {
            if (active[i] < 0.5f) continue;
            const Point3f  p    = position[i];
            const float    wpdf = pdf[i];
            if (!(wpdf > 0.f) || !std::isfinite(wpdf)) continue;
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;

            const float irr = getLuminance(radiance[i]) / wpdf;
            if (!std::isfinite(irr) || irr <= 0.f) continue;

            const size_t leaf = findLeaf(sNodes, m_sTree->m_aabb, p);
            if (!sNodes[leaf].isLeaf) continue;

            Point2f c = DTreeWrapper::dirToCanonical(wo[i]);
            c.x = std::max(0.f, std::min(1.f, c.x));
            c.y = std::max(0.f, std::min(1.f, c.y));

            DTree& b = sNodes[leaf].dTree.building;
            b.m_statisticalWeight += 1.f;
            b.m_sum += irr;

            std::vector<QuadTreeNode>& q = leafNodes[leaf];
            size_t qi = 0;
            while (true) {
                const int ci = q[qi].childIndex(c);
                if (q[qi].isLeaf(ci)) { q[qi].m_sum[ci] += irr; break; }
                qi = q[qi].child(ci);
            }
        }

        // Upload updated trees to device
        for (size_t s = 0; s < numS; ++s) {
            if (!sNodes[s].isLeaf) continue;
            DTree& b = sNodes[s].dTree.building;
            if (b.m_numNodes > 0 && b.m_nodes)
                cudaMemcpy(b.m_nodes, leafNodes[s].data(), b.m_numNodes * sizeof(QuadTreeNode), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(m_sTree->m_nodes, sNodes.data(), numS * sizeof(STreeNode), cudaMemcpyHostToDevice);

        // Refine S-tree and promote building to sampling tree
        const size_t splitThreshold =
            static_cast<size_t>(std::sqrt(std::pow(2.f, static_cast<float>(m_iteration))) * static_cast<float>(m_sTreeThreshold));
        m_sTree->refine(splitThreshold, -1);
        m_sTree->forEachDTreeWrapper([](DTreeWrapper* dw) { dw->build(); });
        m_sTree->forEachDTreeWrapper([this](DTreeWrapper* dw) { dw->reset(20, m_dTreeThreshold); });
        m_iteration++;
        updateIterationLabel();
    }

    void updateLaunchParams(LaunchParams& params) const override {
        params.guiding.active                 = (m_sTree != nullptr);
        params.guiding.type                   = m_sTree ? GuidingType::PPG : GuidingType::None;
        params.guiding.sTreeNodes             = m_sTree ? m_sTree->m_nodes : nullptr;
        params.guiding.sTreeAABB              = m_sTree ? m_sTree->m_aabb : AABB();
        params.guiding.bsdf_sampling_fraction = m_bsdfSamplingFraction;
    }

    void renderUI(nanogui::Widget* parent,
                  std::function<void()> clearFilm,
                  std::function<void()> requestTrain) override {
        using namespace nanogui;

        // Parameters in a 2-column grid so labels/controls line up.
        Widget* grid = new Widget(parent);
        grid->setLayout(new GridLayout(Orientation::Horizontal, 2, Alignment::Middle, 0, 8));

        new Label(grid, "Iteration", "sans-bold");
        m_iterLabel = new Label(grid, std::to_string(m_iteration));

        new Label(grid, "BSDF Fraction", "sans-bold");
        Slider* frac = new Slider(grid);
        frac->setFixedWidth(130);
        frac->setValue(m_bsdfSamplingFraction);
        frac->setCallback([this, clearFilm](float v) { m_bsdfSamplingFraction = v; clearFilm(); });

        new Label(grid, "S-Tree Split Factor", "sans-bold");
        IntBox<int>* split = new IntBox<int>(grid, m_sTreeThreshold);
        split->setFixedWidth(130);
        split->setEditable(true);
        split->setMinMaxValues(1, 100000);
        split->setCallback([this](int v) { m_sTreeThreshold = v; });

        new Label(grid, "D-Tree Threshold", "sans-bold");
        FloatBox<float>* thr = new FloatBox<float>(grid, m_dTreeThreshold);
        thr->setFixedWidth(130);
        thr->setEditable(true);
        thr->setMinMaxValues(0.0001f, 0.5f);
        thr->setCallback([this](float v) { m_dTreeThreshold = v; });

        // Action buttons on their own centered row.
        Widget* buttons = new Widget(parent);
        buttons->setLayout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 6, 6));
        Button* trainBtn = new Button(buttons, "Train Iteration");
        trainBtn->setCallback([requestTrain] { requestTrain(); });
        Button* resetBtn = new Button(buttons, "Reset");
        resetBtn->setCallback([this, clearFilm] { reset(); clearFilm(); });
    }

private:
    static AABB defaultBounds() {
        return AABB(Point3f(-10.f, -10.f, -10.f), Point3f(10.f, 10.f, 10.f));
    }

    static bool valid(const AABB& b) {
        return std::isfinite(b.min.x) && std::isfinite(b.min.y) && std::isfinite(b.min.z) &&
               std::isfinite(b.max.x) && std::isfinite(b.max.y) && std::isfinite(b.max.z) &&
               b.max.x > b.min.x && b.max.y > b.min.y && b.max.z > b.min.z;
    }

    // Descend the host S-tree to the leaf node containing world point p.
    static size_t findLeaf(const std::vector<STreeNode>& nodes, const AABB& aabb, Point3f p) {
        const Vector3f size(aabb.max.x - aabb.min.x, aabb.max.y - aabb.min.y, aabb.max.z - aabb.min.z);
        p.x = std::max(0.f, std::min(1.f, (p.x - aabb.min.x) / size.x));
        p.y = std::max(0.f, std::min(1.f, (p.y - aabb.min.y) / size.y));
        p.z = std::max(0.f, std::min(1.f, (p.z - aabb.min.z) / size.z));

        size_t idx = 0;
        while (!nodes[idx].isLeaf) {
            const int axis = nodes[idx].axis;
            float* pp = (axis == 0) ? &p.x : (axis == 1) ? &p.y : &p.z;
            if (*pp < 0.5f) { *pp *= 2.f;               idx = nodes[idx].children[0]; }
            else            { *pp = (*pp - 0.5f) * 2.f; idx = nodes[idx].children[1]; }
        }
        return idx;
    }

    void rebuild() {
        delete m_sTree;
        m_sTree = new STree(m_bounds);
        m_iteration = 0;
        updateIterationLabel();
    }

    void updateIterationLabel() {
        if (m_iterLabel) m_iterLabel->setCaption(std::to_string(m_iteration));
    }

    STree*          m_sTree = nullptr;
    AABB            m_bounds = defaultBounds();
    int             m_iteration = 0;
    nanogui::Label* m_iterLabel = nullptr;

    float  m_bsdfSamplingFraction = 0.5f;
    int    m_sTreeThreshold = 4000;
    float  m_dTreeThreshold = 0.01f;
};

// ---------------------------------------------------------------------------
// GuidingManager
// ---------------------------------------------------------------------------
GuidingManager::GuidingManager() {
    m_methods.push_back(std::make_unique<NoneGuiding>());
    m_methods.push_back(std::make_unique<PPGGuiding>());
    m_mode = PATH_GUIDING_NONE;
}

GuidingMethod* GuidingManager::active() const {
    for (const auto& m : m_methods)
        if (m->mode() == m_mode) return m.get();
    return m_methods.front().get();
}

void GuidingManager::setSceneBounds(const AABB& bounds) {
    for (const auto& m : m_methods) m->setSceneBounds(bounds);
}

void GuidingManager::reset() { active()->reset(); }

void GuidingManager::train(const TrainingBufferManager& buffers, int maxDepth) {
    active()->train(buffers, maxDepth);
}

void GuidingManager::updateLaunchParams(LaunchParams& params) const {
    active()->updateLaunchParams(params);
}

void GuidingManager::renderUI(nanogui::Widget* grid,
                              nanogui::Widget* panel,
                              std::function<void()> clearFilm,
                              std::function<void()> requestTrain) {
    using namespace nanogui;

    // Method dropdown sits in the settings grid, aligned with the other rows.
    std::vector<std::string> names;
    for (const auto& m : m_methods) names.push_back(m->name());
    ComboBox* combo = new ComboBox(grid, names);
    combo->setFixedWidth(130);

    // One panel per method; only the active one is shown.
    std::vector<Widget*> panels;
    for (const auto& m : m_methods) {
        Widget* p = new Widget(panel);
        p->setLayout(new BoxLayout(Orientation::Vertical, Alignment::Fill, 0, 4));
        m->renderUI(p, clearFilm, requestTrain);
        panels.push_back(p);
    }

    auto syncVisibility = [this, panels]() {
        for (size_t i = 0; i < m_methods.size(); ++i)
            panels[i]->setVisible(m_methods[i]->mode() == m_mode);
    };

    for (size_t i = 0; i < m_methods.size(); ++i)
        if (m_methods[i]->mode() == m_mode) combo->setSelectedIndex(static_cast<int>(i));

    combo->setCallback([this, clearFilm, syncVisibility, panel](int idx) {
        m_mode = m_methods[idx]->mode();
        syncVisibility();
        if (panel->screen()) panel->screen()->performLayout();
        clearFilm();
    });

    syncVisibility();
}

FUTABA_NAMESPACE_END
