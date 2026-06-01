#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "bvh.cuh"
#include "bsdf_sample.cuh"
#include "ray.cuh"
#include "emitter_sampler.cuh"
#include "envmap.cuh"
#include "frame.cuh"
#include "launch_params.h"
#include "path.cuh"
#include "phong.cuh"
#include "surface_interaction.cuh"
#include "volpath.cuh"
#include "material.cuh"
#include "distribution.cuh"
#include "scene.cuh"
#include "perspective.cuh"
#include "sphere.cuh"
#include "triangle.cuh"

#include "ppg/stree.h"
#include "ppg/dtree.cuh"

// Bring types under scope to allow anonymous namespace definitions to pass compiler checks
using namespace futaba;

namespace {

template <typename T>
void print_type(const char* name) {
    std::cout << std::left << std::setw(34) << name
              << " size=" << std::setw(4) << sizeof(T)
              << " align=" << alignof(T) << '\n';
}

float buildRandomNode(std::vector<QuadTreeNode>& hostNodes,
                      size_t nodeIndex,
                      int depthLevel,
                      int maxDepth,
                      std::mt19937& rng,
                      std::uniform_real_distribution<float>& splitChance,
                      std::uniform_real_distribution<float>& leafWeight,
                      int& treeMaxDepth) {
    float nodeTotal = 0.0f;
    const bool forceSplit = (depthLevel == 1) && (maxDepth > 1);
    const float splitProbability = std::clamp(
        0.82f - 0.18f * static_cast<float>(depthLevel - 1),
        0.20f,
        0.82f);
    const bool shouldSplit = depthLevel < maxDepth &&
        (forceSplit || splitChance(rng) < splitProbability);

    for (int quadrant = 0; quadrant < 4; ++quadrant) {
        const float descendProbability = std::clamp(
            0.55f - 0.10f * static_cast<float>(depthLevel - 1),
            0.15f,
            0.55f);

        const bool descend = shouldSplit && depthLevel + 1 < maxDepth &&
            splitChance(rng) < descendProbability;

        if (descend) {
            const size_t childIndex = hostNodes.size();
            hostNodes.emplace_back();
            hostNodes[nodeIndex].setChild(quadrant, static_cast<uint32_t>(childIndex));

            const float childTotal = buildRandomNode(
                hostNodes,
                childIndex,
                depthLevel + 1,
                maxDepth,
                rng,
                splitChance,
                leafWeight,
                treeMaxDepth);

            hostNodes[nodeIndex].setSum(quadrant, childTotal);
            nodeTotal += childTotal;
        } else {
            const float weight = leafWeight(rng);
            hostNodes[nodeIndex].setChild(quadrant, 0);
            hostNodes[nodeIndex].setSum(quadrant, weight);
            nodeTotal += weight;
        }
    }

    treeMaxDepth = std::max(treeMaxDepth, depthLevel);
    return nodeTotal;
}

DTree makeRandomTree(unsigned seed, int maxDepth) {
    std::vector<QuadTreeNode> hostNodes;
    hostNodes.emplace_back();

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> splitChance(0.0f, 1.0f);
    std::uniform_real_distribution<float> leafWeight(0.05f, 1.0f);

    int treeMaxDepth = 1;
    float totalSum = buildRandomNode(hostNodes, 0, 1, maxDepth, rng, splitChance, leafWeight, treeMaxDepth);

    DTree tree;
    tree.clear();
    tree.m_numNodes = hostNodes.size();
    tree.m_maxDepth = treeMaxDepth;
    tree.m_sum = totalSum;
    tree.m_statisticalWeight = 1.0f; 

    cudaMalloc(&tree.m_nodes, tree.m_numNodes * sizeof(QuadTreeNode));
    cudaMemcpy(tree.m_nodes, hostNodes.data(), tree.m_numNodes * sizeof(QuadTreeNode), cudaMemcpyHostToDevice);

    return tree;
}

float sampleTreeValue(const std::vector<QuadTreeNode>& hostNodes, size_t nodeIndex, Point2f sample) {
    const QuadTreeNode& node = hostNodes[nodeIndex];
    Point2f localSample = sample;
    const int quadrant = node.childIndex(localSample);

    if (node.isLeaf(quadrant)) {
        return node.sum(quadrant);
    }

    return sampleTreeValue(hostNodes, node.child(quadrant), localSample);
}

void dumpTreeNode(const std::vector<QuadTreeNode>& hostNodes,
                  size_t nodeIndex,
                  int depthLevel,
                  const std::string& path) {
    const QuadTreeNode& node = hostNodes[nodeIndex];
    const float total = node.sum(0) + node.sum(1) + node.sum(2) + node.sum(3);

    std::cout << std::string(depthLevel * 2, ' ')
              << path
              << "node " << nodeIndex
              << " depth=" << depthLevel
              << " total=" << total
              << " sums=["
              << node.sum(0) << ", "
              << node.sum(1) << ", "
              << node.sum(2) << ", "
              << node.sum(3) << "]\n";

    for (int quadrant = 0; quadrant < 4; ++quadrant) {
        if (!node.isLeaf(quadrant)) {
            dumpTreeNode(hostNodes,
                         node.child(quadrant),
                         depthLevel + 1,
                         path + std::to_string(quadrant) + "/");
        }
    }
}

void printHeatmap(const std::vector<QuadTreeNode>& hostNodes, int width = 48, int height = 24) {
    const std::string ramp = " .:-=+*#%@";
    std::vector<float> values((size_t)width * (size_t)height, 0.0f);
    float maxValue = 0.0f;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const Point2f sample(
                (x + 0.5f) / static_cast<float>(width),
                (y + 0.5f) / static_cast<float>(height));
            const float value = sampleTreeValue(hostNodes, 0, sample);
            values[(size_t)y * (size_t)width + (size_t)x] = value;
            maxValue = std::max(maxValue, value);
        }
    }

    std::cout << "Heatmap (top = v=1, bottom = v=0)\n";
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            const float value = values[(size_t)y * (size_t)width + (size_t)x];
            const float normalized = (maxValue > 0.0f)
                ? std::log1p(value) / std::log1p(maxValue)
                : 0.0f;
            const size_t rampIndex = std::min(
                ramp.size() - 1,
                static_cast<size_t>(normalized * static_cast<float>(ramp.size() - 1) + 0.5f));
            std::cout << ramp[rampIndex];
        }
        std::cout << '\n';
    }
}

void printTreeDebugVisualization() {
    const unsigned seed = 1337u;
    const int maxDepth = 4;
    const DTree tree = makeRandomTree(seed, maxDepth);

    std::vector<QuadTreeNode> hostNodes(tree.numNodes());
    if (tree.numNodes() > 0 && tree.m_nodes != nullptr) {
        cudaMemcpy(hostNodes.data(), tree.m_nodes, hostNodes.size() * sizeof(QuadTreeNode), cudaMemcpyDeviceToHost);
    }

    std::cout << "Random DTree debug visualization\n";
    std::cout << "  seed = " << seed << '\n';
    std::cout << "  nodes = " << tree.numNodes() << '\n';
    std::cout << "  depth = " << tree.depth() << '\n';
    
    if (!hostNodes.empty()) {
        std::cout << "  depthAt(0.5, 0.5) = "
                  << hostNodes[0].depthAt(Point2f(0.5f, 0.5f), hostNodes.data()) << '\n';
    } else {
        std::cout << "  depthAt(0.5, 0.5) = 0\n";
    }

    const std::array<Point2f, 5> probes = {
        Point2f(0.15f, 0.15f),
        Point2f(0.85f, 0.15f),
        Point2f(0.15f, 0.85f),
        Point2f(0.85f, 0.85f),
        Point2f(0.50f, 0.50f)
    };

    std::cout << "Probe depths / values:\n";
    for (const auto& probe : probes) {
        int probeDepth = 0;
        if (!hostNodes.empty()) {
            probeDepth = hostNodes[0].depthAt(probe, hostNodes.data());
        }
        std::cout << "  (" << probe.x << ", " << probe.y << ")"
                  << " depth=" << probeDepth
                  << " value=" << (hostNodes.empty() ? 0.0f : sampleTreeValue(hostNodes, 0, probe)) << '\n';
    }

    std::cout << "\nTree structure:\n";
    if (!hostNodes.empty()) {
        dumpTreeNode(hostNodes, 0, 0, "");
    }

    std::cout << "\nHeatmap:\n";
    if (!hostNodes.empty()) {
        printHeatmap(hostNodes);
    }
}

} // namespace

void printStructSize(){
    print_type<Vector2f>("Vector2f");
    print_type<Point2f>("Point2f");
    print_type<Vector3f>("Vector3f");
    print_type<Point3f>("Point3f");
    print_type<Color3f>("Color3f");
    print_type<Normal3f>("Normal3f");

    print_type<Ray3f>("Ray3f");
    print_type<Frame>("Frame");
    print_type<BSDFSample>("BSDFSample");
    print_type<Material>("Material");
    print_type<SurfaceIntersection>("SurfaceIntersection");
    print_type<Triangle>("Triangle");
    print_type<Sphere>("Sphere");
    print_type<AABB>("AABB");
    print_type<BVHNode>("BVHNode");
    print_type<BVH>("BVH");

    print_type<EnvironmentMapEmitter>("EnvironmentMapEmitter");
    print_type<EmitterSample>("EmitterSample");
    print_type<EmitterGPU>("EmitterGPU");
    print_type<Scene>("Scene");
    print_type<PerspectiveCamera>("PerspectiveCamera");

    print_type<Distribution1D>("Distribution1D");
    print_type<Distribution2D>("Distribution2D");
    print_type<Path>("Path");
    print_type<VolumetricPath>("VolumetricPath");
    print_type<Phong>("Phong");
    print_type<LaunchParams>("LaunchParams");
    print_type<QuadTreeNode>("QuadTreeNode");
    print_type<DTree>("DTree");
}

int main(int argc, char** argv) {
    bool showTree = true;
    bool showSizes = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--sizes") {
            showTree = false;
            showSizes = true;
        } else if (arg == "--both") {
            showTree = true;
            showSizes = true;
        } else if (arg == "--tree") {
            showTree = true;
        }
    }

    if (showTree) {
        printTreeDebugVisualization();
    }

    if (showSizes) {
        std::cout << "\nType sizes:\n";
        printStructSize();
    }

    return 0;
}