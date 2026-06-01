#include "guiding.h"
#include "launch_params.h"
#include "hdrfilm.cuh"
#include <iostream>
#include <cmath>
#include <nanogui/nanogui.h>

namespace futaba {

class NoneGuidingMethod : public GuidingMethod {
public:
    std::string getName() const override { return "None"; }
    int getMode() const override { return PATH_GUIDING_NONE; }
    std::vector<std::string> getVisualizerBuffers() const override { return {"Active"}; }
};

static void boxBlur(const std::vector<Color3f>& input, std::vector<Color3f>& output, int width, int height, int radius) {
    std::vector<Color3f> temp(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Color3f sum(0.f);
            float count = 0.f;
            for (int dx = -radius; dx <= radius; ++dx) {
                int nx = x + dx;
                if (nx >= 0 && nx < width) {
                    sum += input[y * width + nx];
                    count += 1.f;
                }
            }
            temp[y * width + x] = sum / count;
        }
    }
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Color3f sum(0.f);
            float count = 0.f;
            for (int dy = -radius; dy <= radius; ++dy) {
                int ny = y + dy;
                if (ny >= 0 && ny < height) {
                    sum += temp[ny * width + x];
                    count += 1.f;
                }
            }
            output[y * width + x] = sum / count;
        }
    }
}

class PPGGuidingMethod : public GuidingMethod {
public:
    PPGGuidingMethod() : m_sTree(nullptr), m_iteration(0), m_width(0), m_height(0),
                     m_bsdfSamplingFraction(0.5f), m_sTreeThreshold(4000), m_dTreeThreshold(0.01f),
                     m_spatialFilter(0), m_directionalFilter(0), m_distributionMode(0) {}
    ~PPGGuidingMethod() { destroy(); }

    std::string getName() const override { return "SD-Tree (PPG)"; }
    int getMode() const override { return PATH_GUIDING_PPG; }

    void initialize(int width, int height) override {
        m_width = width;
        m_height = height;
        destroy();
        m_sTree = new STree(defaultBounds());
    }

    void resize(int width, int height) override { m_width = width; m_height = height; }
    void setSceneBounds(const AABB& bounds) override {
        m_sceneBounds = validBounds(bounds) ? bounds : defaultBounds();
        destroy();
        m_sTree = new STree(m_sceneBounds);
        m_iteration = 0;
    }
    void reset() override {
        if (m_sTree) {
            AABB bounds = m_sTree->m_aabb;
            destroy();
            m_sTree = new STree(bounds);
        }
        m_iteration = 0;
    }

    void train(const TrainingBufferManager& bufferManager, int maxDepth, const HDRFilm* film = nullptr) override {
        if (!m_sTree || m_width == 0 || m_height == 0 || maxDepth == 0) return;
        if (!bufferManager.getActive() || !bufferManager.getPosition() || !bufferManager.getRadiance() ||
            !bufferManager.getWo() || !bufferManager.getDirectionPdf()) return;
        if (m_distributionMode == 3 && !bufferManager.getWi()) return;

        size_t totalElements = static_cast<size_t>(m_width) * m_height * maxDepth;
        
        std::vector<float> hostActive(totalElements);
        std::vector<Point3f> hostPosition(totalElements);
        std::vector<Color3f> hostRadiance(totalElements);
        std::vector<Color3f> hostWo(totalElements);
        std::vector<float> hostPdf(totalElements);
        std::vector<Color3f> hostWi;

        cudaMemcpy(hostActive.data(), bufferManager.getActive(), totalElements * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostPosition.data(), bufferManager.getPosition(), totalElements * sizeof(Point3f), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostRadiance.data(), bufferManager.getRadiance(), totalElements * sizeof(Color3f), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostWo.data(), bufferManager.getWo(), totalElements * sizeof(Color3f), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostPdf.data(), bufferManager.getDirectionPdf(), totalElements * sizeof(float), cudaMemcpyDeviceToHost);
        if (m_distributionMode == 3) {
            hostWi.resize(totalElements);
            cudaMemcpy(hostWi.data(), bufferManager.getWi(), totalElements * sizeof(Color3f), cudaMemcpyDeviceToHost);
        }
        cudaDeviceSynchronize();

        // 1. If Product (L2 Full), blur the film
        std::vector<Color3f> blurred1(m_width * m_height);
        std::vector<Color3f> blurred2(m_width * m_height);
        if (m_distributionMode == 3 && film != nullptr) {
            std::vector<Color3f> hostFilm(m_width * m_height);
            cudaMemcpy(hostFilm.data(), film->d_pixels, m_width * m_height * sizeof(Color3f), cudaMemcpyDeviceToHost);
            float scale = film->sampleCount > 0 ? 1.f / film->sampleCount : 1.f;
            for (auto& p : hostFilm) {
                p *= scale;
            }
            boxBlur(hostFilm, blurred1, m_width, m_height, 3);
            boxBlur(blurred1, blurred2, m_width, m_height, 3);
        }

        size_t numSNodes = m_sTree->m_numNodes;
        if (numSNodes == 0) return;

        std::vector<STreeNode> hostSNodes(numSNodes);
        cudaMemcpy(hostSNodes.data(), m_sTree->m_nodes, numSNodes * sizeof(STreeNode), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        struct LeafDTreeHost {
            size_t sNodeIdx;
            std::vector<QuadTreeNode> hostBuildingNodes;
            std::vector<QuadTreeNode> hostSamplingNodes;
        };
        std::vector<LeafDTreeHost> leafDTrees;
        std::vector<int> sNodeToLeafIdx(numSNodes, -1);

        for (size_t si = 0; si < numSNodes; ++si) {
            if (!hostSNodes[si].isLeaf) continue;
            DTreeWrapper& dw = hostSNodes[si].dTree;
            DTree& bld = dw.building;
            DTree& smp = dw.sampling;
            
            LeafDTreeHost lh;
            lh.sNodeIdx = si;
            if (bld.m_numNodes > 0 && bld.m_nodes != nullptr) {
                lh.hostBuildingNodes.resize(bld.m_numNodes);
                cudaMemcpy(lh.hostBuildingNodes.data(), bld.m_nodes, bld.m_numNodes * sizeof(QuadTreeNode), cudaMemcpyDeviceToHost);
            } else {
                lh.hostBuildingNodes.resize(1);
            }
            if (smp.m_numNodes > 0 && smp.m_nodes != nullptr) {
                lh.hostSamplingNodes.resize(smp.m_numNodes);
                cudaMemcpy(lh.hostSamplingNodes.data(), smp.m_nodes, smp.m_numNodes * sizeof(QuadTreeNode), cudaMemcpyDeviceToHost);
            } else {
                lh.hostSamplingNodes.resize(1);
            }
            sNodeToLeafIdx[si] = (int)leafDTrees.size();
            leafDTrees.push_back(std::move(lh));
        }
        cudaDeviceSynchronize();

        // Helper maps location and extracts specific voxel dimension sizing
        auto findLeafIdxAndSize = [&](Point3f p, Vector3f& outVoxelSize) -> size_t {
            Vector3f sTreeSize(m_sTree->m_aabb.maxP.x - m_sTree->m_aabb.minP.x,
                               m_sTree->m_aabb.maxP.y - m_sTree->m_aabb.minP.y,
                               m_sTree->m_aabb.maxP.z - m_sTree->m_aabb.minP.z);
            outVoxelSize = sTreeSize;
            p.x = (p.x - m_sTree->m_aabb.minP.x) / sTreeSize.x;
            p.y = (p.y - m_sTree->m_aabb.minP.y) / sTreeSize.y;
            p.z = (p.z - m_sTree->m_aabb.minP.z) / sTreeSize.z;
            
            p.x = std::max(0.f, std::min(1.f, p.x));
            p.y = std::max(0.f, std::min(1.f, p.y));
            p.z = std::max(0.f, std::min(1.f, p.z));

            size_t idx = 0;
            while (!hostSNodes[idx].isLeaf) {
                int axis = hostSNodes[idx].axis;
                outVoxelSize[axis] *= 0.5f;
                float* pp = (axis == 0) ? &p.x : (axis == 1) ? &p.y : &p.z;
                if (*pp < 0.5f) {
                    *pp *= 2.f;
                    idx = hostSNodes[idx].children[0];
                } else {
                    *pp = (*pp - 0.5f) * 2.f;
                    idx = hostSNodes[idx].children[1];
                }
            }
            return idx;
        };

        // Helper to compute depth of canonDir in host quadtree
        auto getDTreeDepth = [](const Point2f& p, const std::vector<QuadTreeNode>& qNodes) -> int {
            Point2f cp = p;
            size_t qIdx = 0;
            int d = 1;
            while (true) {
                int ci = qNodes[qIdx].childIndex(cp);
                if (qNodes[qIdx].isLeaf(ci)) return d;
                qIdx = qNodes[qIdx].child(ci);
                d++;
            }
        };

        // Helper to record directional contribution stochastically/box-wise into host quadtree
        auto recordBoxDir = [](auto& self, size_t qIdx, const Point2f& origin, float size, Point2f nodeOrigin, float nodeSize, float value, std::vector<QuadTreeNode>& qNodes) -> void {
            float childSize = nodeSize / 2.0f;
            for (int i = 0; i < 4; ++i) {
                Point2f childOrigin = nodeOrigin;
                if (i & 1) { childOrigin.x += childSize; }
                if (i & 2) { childOrigin.y += childSize; }

                float xOverlap = std::max(0.0f, std::min(origin.x + size, childOrigin.x + childSize) - std::max(origin.x, childOrigin.x));
                float yOverlap = std::max(0.0f, std::min(origin.y + size, childOrigin.y + childSize) - std::max(origin.y, childOrigin.y));
                float w = xOverlap * yOverlap;

                if (w > 0.0f) {
                    if (qNodes[qIdx].isLeaf(i)) {
                        qNodes[qIdx].m_sum[i] += value * w;
                    } else {
                        self(self, qNodes[qIdx].child(i), origin, size, childOrigin, childSize, value, qNodes);
                    }
                }
            }
        };

        // Helper to recursively record spatial contribution box-wise into host S-Tree/D-Trees
        auto recordSpatialBox = [&](auto& self, size_t sIdx, const Point3f& min1, const Point3f& max1, Point3f min2, Vector3f size2, float irradiance, float statisticalWeight, const Point2f& canonDir, EDirectionalFilter dirFilter) -> void {
            float w = 1.0f;
            for (int i = 0; i < 3; ++i) {
                float l = std::max(0.0f, std::min(max1[i], min2[i] + size2[i]) - std::max(min1[i], min2[i]));
                w *= l;
            }
            if (w <= 0.0f) return;

            if (hostSNodes[sIdx].isLeaf) {
                int leafIdx = sNodeToLeafIdx[sIdx];
                if (leafIdx < 0) return;
                LeafDTreeHost& lh = leafDTrees[leafIdx];
                DTreeWrapper& dw = hostSNodes[sIdx].dTree;
                
                float wWeight = statisticalWeight * w;
                dw.building.m_statisticalWeight += wWeight;

                if (irradiance > 0.f) {
                    float wIrr = irradiance * wWeight;
                    dw.building.m_sum += wIrr;

                    if (dirFilter == EDirectionalFilter::ENearest) {
                        QuadTreeNode* hostQNodes = lh.hostBuildingNodes.data();
                        size_t qIdx = 0;
                        Point2f cp = canonDir;
                        while (true) {
                            int ci = hostQNodes[qIdx].childIndex(cp);
                            if (hostQNodes[qIdx].isLeaf(ci)) {
                                hostQNodes[qIdx].m_sum[ci] += wIrr;
                                break;
                            } else {
                                qIdx = hostQNodes[qIdx].child(ci);
                            }
                        }
                    } else {
                        int d = getDTreeDepth(canonDir, lh.hostBuildingNodes);
                        float dSize = std::pow(0.5f, d);
                        Point2f origin(canonDir.x - dSize * 0.5f, canonDir.y - dSize * 0.5f);
                        float value = wIrr / (dSize * dSize);
                        recordBoxDir(recordBoxDir, 0, origin, dSize, Point2f(0.f, 0.f), 1.f, value, lh.hostBuildingNodes);
                    }
                }
            } else {
                int axis = hostSNodes[sIdx].axis;
                size2[axis] *= 0.5f;
                for (int i = 0; i < 2; ++i) {
                    Point3f childMin = min2;
                    if (i == 1) childMin[axis] += size2[axis];
                    self(self, hostSNodes[sIdx].children[i], min1, max1, childMin, size2, irradiance, statisticalWeight, canonDir, dirFilter);
                }
            }
        };

        // Thread safe local context LCG pseudo-sampler
        unsigned int lcgSeed = 7193113u;
        auto hostRng = [&]() {
            lcgSeed = lcgSeed * 1664525u + 1013904223u;
            return (float)(lcgSeed & 0xFFFFFF) / 16777216.0f;
        };

        for (size_t i = 0; i < totalElements; ++i) {
            if (hostActive[i] <= 0.5f) continue;

            float lum = 0.2126f * hostRadiance[i].x + 0.7152f * hostRadiance[i].y + 0.0722f * hostRadiance[i].z;
            if (!std::isfinite(lum) || lum < 0.f) lum = 0.f;
            const float directionPdf = hostPdf[i];
            if (!std::isfinite(directionPdf) || directionPdf <= 0.f) continue;

            Vector3f wo = hostWo[i];
            if (!std::isfinite(wo.x) || !std::isfinite(wo.y) || !std::isfinite(wo.z)) continue;

            Point3f pos = hostPosition[i];
            if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z)) continue;

            Vector3f initialVoxelSize;
            size_t baseLeafSIdx = findLeafIdxAndSize(pos, initialVoxelSize);
            
            Point2f canonDir = DTreeWrapper::dirToCanonical(wo);
            canonDir.x = std::max(0.f, std::min(1.f, canonDir.x));
            canonDir.y = std::max(0.f, std::min(1.f, canonDir.y));

            float irradiance = 0.f;
            if (m_distributionMode == 3) {
                int pixel_idx = i % (m_width * m_height);
                Color3f est = (film != nullptr) ? blurred2[pixel_idx] : Color3f(1.f);
                est.x = std::max(est.x, 1e-3f);
                est.y = std::max(est.y, 1e-3f);
                est.z = std::max(est.z, 1e-3f);

                Color3f outgoing = hostRadiance[i];
                Color3f beta = hostWi[i];

                int baseLeafIdx = sNodeToLeafIdx[baseLeafSIdx];
                float dTreePdf = 1.0f / (4.0f * M_PI);
                if (baseLeafIdx >= 0) {
                    LeafDTreeHost& lh = leafDTrees[baseLeafIdx];
                    DTreeWrapper& dw = hostSNodes[lh.sNodeIdx].dTree;
                    DTree& smp = dw.sampling;

                    QuadTreeNode* devNodes = smp.m_nodes;
                    smp.m_nodes = lh.hostSamplingNodes.data();
                    dTreePdf = smp.pdf(canonDir);
                    smp.m_nodes = devNodes;
                }

                Color3f ratio = (outgoing * beta) / est;
                float ratio_mean = (ratio.x + ratio.y + ratio.z) / 3.0f;
                float val = ratio_mean * dTreePdf;
                irradiance = (val * val) / directionPdf;
            } else {
                irradiance = lum / directionPdf;
            }

            if (m_spatialFilter == 2) { // True Box filter
                float volume = initialVoxelSize.x * initialVoxelSize.y * initialVoxelSize.z;
                if (volume > 0.f) {
                    Point3f half(initialVoxelSize.x * 0.5f, initialVoxelSize.y * 0.5f, initialVoxelSize.z * 0.5f);
                    Point3f min1(pos.x - half.x, pos.y - half.y, pos.z - half.z);
                    Point3f max1(pos.x + half.x, pos.y + half.y, pos.z + half.z);
                    Vector3f sTreeSize(m_sTree->m_aabb.maxP.x - m_sTree->m_aabb.minP.x,
                                       m_sTree->m_aabb.maxP.y - m_sTree->m_aabb.minP.y,
                                       m_sTree->m_aabb.maxP.z - m_sTree->m_aabb.minP.z);
                    
                    recordSpatialBox(recordSpatialBox, 0, min1, max1, m_sTree->m_aabb.minP, sTreeSize,
                                     irradiance, 1.0f / volume, canonDir, (EDirectionalFilter)m_directionalFilter);
                }
            } else {
                if (m_spatialFilter == 1) { // StochasticBox
                    pos.x += initialVoxelSize.x * (hostRng() - 0.5f);
                    pos.y += initialVoxelSize.y * (hostRng() - 0.5f);
                    pos.z += initialVoxelSize.z * (hostRng() - 0.5f);
                }

                size_t leafSIdx = findLeafIdxAndSize(pos, initialVoxelSize);
                int leafIdx = sNodeToLeafIdx[leafSIdx];
                if (leafIdx < 0) continue;

                LeafDTreeHost& lh = leafDTrees[leafIdx];
                DTreeWrapper& dw = hostSNodes[lh.sNodeIdx].dTree;
                
                float statisticalWeight = 1.0f;
                dw.building.m_statisticalWeight += statisticalWeight;

                if (std::isfinite(irradiance) && irradiance > 0.f) {
                    dw.building.m_sum += irradiance * statisticalWeight;

                    if (m_directionalFilter == 0) { // Nearest
                        QuadTreeNode* hostQNodes = lh.hostBuildingNodes.data();
                        size_t qIdx = 0;
                        while (true) {
                            int ci = hostQNodes[qIdx].childIndex(canonDir);
                            if (hostQNodes[qIdx].isLeaf(ci)) {
                                hostQNodes[qIdx].m_sum[ci] += irradiance * statisticalWeight;
                                break;
                            } else {
                                qIdx = hostQNodes[qIdx].child(ci);
                            }
                        }
                    } else { // Box
                        int d = getDTreeDepth(canonDir, lh.hostBuildingNodes);
                        float dSize = std::pow(0.5f, d);
                        Point2f origin(canonDir.x - dSize * 0.5f, canonDir.y - dSize * 0.5f);
                        float value = irradiance * statisticalWeight / (dSize * dSize);
                        recordBoxDir(recordBoxDir, 0, origin, dSize, Point2f(0.f, 0.f), 1.f, value, lh.hostBuildingNodes);
                    }
                }
            }
        }

        for (auto& lh : leafDTrees) {
            DTreeWrapper& dw = hostSNodes[lh.sNodeIdx].dTree;
            if (dw.building.m_numNodes > 0 && dw.building.m_nodes != nullptr) {
                cudaMemcpy(dw.building.m_nodes, lh.hostBuildingNodes.data(), dw.building.m_numNodes * sizeof(QuadTreeNode), cudaMemcpyHostToDevice);
            }
        }

        cudaMemcpy(m_sTree->m_nodes, hostSNodes.data(), numSNodes * sizeof(STreeNode), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        // 8. Execute refinement passes with structural data loaded properly
        int spp = (film != nullptr && film->sampleCount > 0) ? film->sampleCount : 16;
        size_t splitThreshold = static_cast<size_t>(std::sqrt(std::pow(2, m_iteration) * spp / 4.f) * (float)m_sTreeThreshold);
        m_sTree->refine(splitThreshold, -1); 

        m_sTree->forEachDTreeWrapper([this](DTreeWrapper* dTree) { dTree->build(m_distributionMode); });
    
        std::cout << "\n=== [Training Iteration " << m_iteration << " Completed] ===" << std::endl;
        m_sTree->gatherStatistics();
        std::cout << "================================================\n" << std::endl;

        m_sTree->forEachDTreeWrapper([this](DTreeWrapper* dTree) { dTree->reset(20, m_dTreeThreshold); });
        m_iteration++;
    }

    void postprocess() override {}

    std::vector<std::string> getVisualizerBuffers() const override {
        return {"Active", "Position", "Outgoing Angle (wo)", "Incoming Radiance"};
    }

    void updateLaunchParams(LaunchParams& params) const override {
        params.sTreeNodes = m_sTree ? m_sTree->m_nodes : nullptr;
        params.sTreeAABB = m_sTree ? m_sTree->m_aabb : AABB();
        params.bsdf_sampling_fraction = m_bsdfSamplingFraction;
        params.ppg_distribution_mode = m_distributionMode;
    }

    void renderUI(nanogui::Widget* parent, std::function<void()> clearFilm, std::function<void()> trainTree) override {
        using namespace nanogui;

        Widget* iterPanel = new Widget(parent);
        iterPanel->setLayout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 10));
        new Label(iterPanel, "SD-Tree Iteration:", "sans-bold");
        auto* iterValLabel = new Label(iterPanel, std::to_string(m_iteration));

        Widget* trainPanel = new Widget(parent);
        trainPanel->setLayout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 5));

        auto* trainBtn = new Button(trainPanel, "Train Iteration");
        trainBtn->setCallback([this, clearFilm, trainTree, iterValLabel]() {
            trainTree();
            clearFilm();
            iterValLabel->setCaption(std::to_string(m_iteration));
        });

        auto* resetBtn = new Button(trainPanel, "Reset Guiding");
        resetBtn->setCallback([this, clearFilm, iterValLabel]() {
            reset();
            clearFilm();
            iterValLabel->setCaption(std::to_string(m_iteration));
        });

        Widget* grid = new Widget(parent);
        grid->setLayout(new GroupLayout(10, 5, 5, 5));

        new Label(grid, "BSDF Fraction", "sans-bold");
        Widget* fractionPanel = new Widget(grid);
        fractionPanel->setLayout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 10));
        auto* fractionSlider = new Slider(fractionPanel);
        fractionSlider->setValue(m_bsdfSamplingFraction);
        fractionSlider->setFixedWidth(100);
        auto* fractionBox = new FloatBox<float>(fractionPanel, m_bsdfSamplingFraction);
        fractionBox->setFixedWidth(50);
        fractionSlider->setCallback([this, clearFilm, fractionBox](float value) {
            m_bsdfSamplingFraction = value;
            fractionBox->setValue(value);
            clearFilm();
        });

        new Label(grid, "S-Tree Split Factor", "sans-bold");
        Widget* sTreePanel = new Widget(grid);
        sTreePanel->setLayout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 10));
        auto* sTreeBox = new IntBox<int>(sTreePanel, m_sTreeThreshold);
        sTreeBox->setEditable(true);
        sTreeBox->setSpinnable(true);
        sTreeBox->setMinMaxValues(1, 100000);
        sTreeBox->setFixedWidth(80);
        sTreeBox->setCallback([this](int value) { m_sTreeThreshold = value; });

        new Label(grid, "D-Tree Threshold", "sans-bold");
        Widget* dTreePanel = new Widget(grid);
        dTreePanel->setLayout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 10));
        auto* dTreeBox = new FloatBox<float>(dTreePanel, m_dTreeThreshold);
        dTreeBox->setEditable(true);
        dTreeBox->setSpinnable(true);
        dTreeBox->setMinMaxValues(0.0001f, 0.5f);
        dTreeBox->setFixedWidth(80);
        dTreeBox->setCallback([this](float value) { m_dTreeThreshold = value; });

        new Label(grid, "Spatial Filter", "sans-bold");
        auto* spatialCombo = new ComboBox(grid, {"Nearest", "StochasticBox", "Box"});
        spatialCombo->setSelectedIndex(m_spatialFilter);
        spatialCombo->setFixedWidth(130);
        spatialCombo->setCallback([this](int index) { m_spatialFilter = index; });

        new Label(grid, "Directional Filter", "sans-bold");
        auto* directionalCombo = new ComboBox(grid, {"Nearest", "Box"});
        directionalCombo->setSelectedIndex(m_directionalFilter);
        directionalCombo->setFixedWidth(130);
        directionalCombo->setCallback([this](int index) { m_directionalFilter = index; });

        new Label(grid, "Target Distribution", "sans-bold");
        auto* distributionCombo = new ComboBox(grid, {"Radiance", "Product (L1)", "Product (L2)", "Product (L2 Full)"});
        distributionCombo->setSelectedIndex(m_distributionMode);
        distributionCombo->setFixedWidth(130);
        distributionCombo->setCallback([this](int index) { m_distributionMode = index; });
    }

private:
    static AABB defaultBounds() {
        AABB bounds;
        bounds.minP = Point3f(-10.f, -10.f, -10.f);
        bounds.maxP = Point3f(10.f, 10.f, 10.f);
        return bounds;
    }

    static bool validBounds(const AABB& bounds) {
        return std::isfinite(bounds.minP.x) && std::isfinite(bounds.minP.y) && std::isfinite(bounds.minP.z) &&
               std::isfinite(bounds.maxP.x) && std::isfinite(bounds.maxP.y) && std::isfinite(bounds.maxP.z) &&
               bounds.maxP.x > bounds.minP.x && bounds.maxP.y > bounds.minP.y && bounds.maxP.z > bounds.minP.z;
    }

    void destroy() { if (m_sTree) { delete m_sTree; m_sTree = nullptr; } }
    STree* m_sTree;
    AABB m_sceneBounds = defaultBounds();
    int m_iteration;
    int m_width;
    int m_height;
    float m_bsdfSamplingFraction;
    int m_sTreeThreshold;
    float m_dTreeThreshold;
    int m_spatialFilter;
    int m_directionalFilter;
    int m_distributionMode;
};

class NPMGuidingMethod : public GuidingMethod {
public:
    std::string getName() const override { return "NPM (Neural)"; }
    int getMode() const override { return PATH_GUIDING_NPM; }
    std::vector<std::string> getVisualizerBuffers() const override {
        return {"Active", "Position", "Normals", "Incoming Angle (wi)", "Outgoing Angle (wo)", "Incoming Radiance", "Material ID"};
    }
};

std::vector<std::shared_ptr<GuidingMethod>>& GuidingRegistry::getMethods() {
    static std::vector<std::shared_ptr<GuidingMethod>> methods;
    if (methods.empty()) {
        methods.push_back(std::make_shared<NoneGuidingMethod>());
        methods.push_back(std::make_shared<PPGGuidingMethod>());
        methods.push_back(std::make_shared<NPMGuidingMethod>());
    }
    return methods;
}

std::shared_ptr<GuidingMethod> GuidingRegistry::getMethod(int mode) {
    for (const auto& method : getMethods()) { if (method->getMode() == mode) return method; }
    return nullptr;
}

GuidingManager::GuidingManager() : m_mode(PATH_GUIDING_NONE), m_width(0), m_height(0) {}
GuidingManager::~GuidingManager() { destroy(); }

bool GuidingManager::init(int width, int height) {
    m_width = width; m_height = height;
    for (auto& method : GuidingRegistry::getMethods()) { method->initialize(width, height); }
    return true;
}

void GuidingManager::resize(int width, int height) {
    m_width = width; m_height = height;
    for (auto& method : GuidingRegistry::getMethods()) { method->resize(width, height); }
}

void GuidingManager::setSceneBounds(const AABB& bounds) {
    for (auto& method : GuidingRegistry::getMethods()) { method->setSceneBounds(bounds); }
}

void GuidingManager::reset() { auto method = GuidingRegistry::getMethod(m_mode); if (method) method->reset(); }
void GuidingManager::train(const TrainingBufferManager& bufferManager, int maxDepth, const HDRFilm* film) {
    auto method = GuidingRegistry::getMethod(m_mode); if (method) method->train(bufferManager, maxDepth, film);
}
void GuidingManager::preprocess() { auto method = GuidingRegistry::getMethod(m_mode); if (method) method->preprocess(); }
void GuidingManager::postprocess() { auto method = GuidingRegistry::getMethod(m_mode); if (method) method->postprocess(); }
void GuidingManager::destroy() {}

} // namespace futaba
