#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stack>
#include <vector>

#include "ppg/dtree.cuh"
#include "bvh.cuh"

enum class ESpatialFilter {
    ENearest,
    EStochasticBox,
    EBox,
};

namespace futaba {

struct STreeNode {
    bool        isLeaf      = true;
    int         axis        = 0;       
    uint32_t    children[2] = {0, 0};  
    DTreeWrapper dTree;                

    // Explicitly clean compiler defaulted initializers
    STreeNode() = default;

    inline __host__ __device__ int childIndex(Point3f& p) const {
        if (p[axis] < 0.5f) {
            p[axis] *= 2.f;
            return 0;
        } else {
            p[axis] = (p[axis] - 0.5f) * 2.f;
            return 1;
        }
    }

    inline __host__ __device__ int nodeIndex(Point3f& p) const { return children[childIndex(p)]; }
    inline __host__ __device__ DTreeWrapper* dTreeWrapper(Point3f& p, Vector3f& size, STreeNode* nodes) {
        STreeNode* curr = this;
        while (!curr->isLeaf) {
            size[curr->axis] *= 0.5f;
            curr = &nodes[curr->nodeIndex(p)];
        }
        return &curr->dTree;
    }

    inline __host__ __device__ const DTreeWrapper* dTreeWrapper() const { return &dTree; }
#ifndef __CUDA_ARCH__
    inline __host__ __device__ int depth(Point3f p, const STreeNode* nodes) const {
        if (isLeaf) return 1;
        return 1 + nodes[nodeIndex(p)].depth(p, nodes);
    }

    inline __host__ __device__ int depth(const STreeNode* nodes) const {
        if (isLeaf) return 1;
        return 1 + safe_max(nodes[children[0]].depth(nodes), nodes[children[1]].depth(nodes));
    }
#endif

#ifndef __CUDA_ARCH__
    inline __host__ void forEachLeaf(
        std::function<void(const DTreeWrapper*, const Point3f&, const Vector3f&)> func,
        Point3f p, Vector3f size, const STreeNode* nodes) const {
        if (isLeaf) {
            func(&dTree, p, size);
        } else {
            size[axis] *= 0.5f;
            for (int i = 0; i < 2; ++i) {
                Point3f childP = p;
                if (i == 1) childP[axis] += size[axis];
                nodes[children[i]].forEachLeaf(func, childP, size, nodes);
            }
        }
    }
#endif

    inline __host__ __device__ static float computeOverlappingVolume(const Point3f& min1, const Point3f& max1,
                                                                     const Point3f& min2, const Point3f& max2) {
        float v = 1.f;
        for (int i = 0; i < 3; ++i)
            v *= safe_max(0.f, safe_min(max1[i], max2[i]) - safe_max(min1[i], min2[i]));
        return v;
    }

#ifndef __CUDA_ARCH__
    inline __host__ __device__ void record(const Point3f& min1, const Point3f& max1,
                                           Point3f min2, Vector3f size2,
                                           const DTreeRecord& rec,
                                           EDirectionalFilter dirFilter,
                                           STreeNode* nodes) {
        const float w = computeOverlappingVolume(min1, max1, min2, Point3f(min2.x + size2.x, min2.y + size2.y, min2.z + size2.z));
        if (w <= 0.f) return;

        if (isLeaf) {
            DTreeRecord wr = rec;
            wr.statisticalWeight *= w;
            dTree.record(wr, dirFilter);
        } else {
            size2[axis] *= 0.5f;
            for (int i = 0; i < 2; ++i) {
                Point3f childMin = min2;
                if (i == 1) childMin[axis] += size2[axis];
                nodes[children[i]].record(min1, max1, childMin, size2, rec, dirFilter, nodes);
            }
        }
    }
#endif
};

struct STree {
    STreeNode* m_nodes; 
    size_t     m_numNodes;
    AABB       m_aabb;

    // Added explicit dual-space viewing constructor to solve stack allocation issues
    inline __host__ __device__ STree() : m_nodes(nullptr), m_numNodes(0) {}

#ifndef __CUDA_ARCH__
    __host__ void initialize() {
        m_numNodes = 1;
        std::vector<STreeNode> rootNode(1);
        cudaMalloc(&m_nodes, sizeof(STreeNode));
        cudaMemcpy(m_nodes, rootNode.data(), sizeof(STreeNode), cudaMemcpyHostToDevice);
    }

    explicit __host__ STree(const AABB& aabb) : m_nodes(nullptr), m_numNodes(0) {
        m_aabb = aabb;
        Vector3f size = m_aabb.maxP - m_aabb.minP;
        float maxSize = std::max({size.x, size.y, size.z});
        m_aabb.maxP = Point3f(m_aabb.minP.x + maxSize, m_aabb.minP.y + maxSize, m_aabb.minP.z + maxSize);
        initialize();
    }

    __host__ void clear() {
        if (m_nodes) {
            std::vector<STreeNode> hostNodes(m_numNodes);
            cudaMemcpy(hostNodes.data(), m_nodes, m_numNodes * sizeof(STreeNode), cudaMemcpyDeviceToHost);
            for (auto& node : hostNodes) {
                if (node.isLeaf) {
                    node.dTree.building.clear();
                    node.dTree.sampling.clear();
                }
            }
            cudaFree(m_nodes);
            m_nodes = nullptr;
        }
        m_numNodes = 0;
    }

    __host__ ~STree() { clear(); }

    __host__ STree& operator=(const STree& other) {
        if (this != &other) {
            clear();
            m_numNodes = other.m_numNodes;
            m_aabb = other.m_aabb;
            if (m_numNodes > 0 && other.m_nodes != nullptr) {
                cudaMalloc(&m_nodes, m_numNodes * sizeof(STreeNode));
                cudaMemcpy(m_nodes, other.m_nodes, m_numNodes * sizeof(STreeNode), cudaMemcpyDeviceToDevice);
            }
        }
        return *this;
    }

    __host__ STree(const STree& other) : m_nodes(nullptr), m_numNodes(0) { *this = other; }
#endif

    inline __host__ __device__ DTreeWrapper* dTreeWrapper(Point3f p, Vector3f& size) {
        size = Vector3f(m_aabb.maxP.x - m_aabb.minP.x, m_aabb.maxP.y - m_aabb.minP.y, m_aabb.maxP.z - m_aabb.minP.z);
        p.x = (p.x - m_aabb.minP.x) / size.x;
        p.y = (p.y - m_aabb.minP.y) / size.y;
        p.z = (p.z - m_aabb.minP.z) / size.z;
        return m_nodes[0].dTreeWrapper(p, size, m_nodes);
    }

    inline __host__ __device__ DTreeWrapper* dTreeWrapper(Point3f p) { Vector3f size; return dTreeWrapper(p, size); }

#ifndef __CUDA_ARCH__
    inline __host__ __device__ void record(const Point3f& p, const DTreeRecord& rec, EDirectionalFilter dirFilter) {
        DTreeWrapper* dw = dTreeWrapper(p);
        if (dw) dw->record(rec, dirFilter);
    }

    inline __host__ __device__ void record(const Point3f& p, const Vector3f& dTreeVoxelSize, DTreeRecord rec, EDirectionalFilter dirFilter) {
        float volume = dTreeVoxelSize.x * dTreeVoxelSize.y * dTreeVoxelSize.z;
        if (volume <= 0.f) return;
        rec.statisticalWeight /= volume;

        Vector3f extents(m_aabb.maxP.x - m_aabb.minP.x, m_aabb.maxP.y - m_aabb.minP.y, m_aabb.maxP.z - m_aabb.minP.z);
        Point3f half(dTreeVoxelSize.x * 0.5f, dTreeVoxelSize.y * 0.5f, dTreeVoxelSize.z * 0.5f);

        m_nodes[0].record(
            Point3f(p.x - half.x, p.y - half.y, p.z - half.z),
            Point3f(p.x + half.x, p.y + half.y, p.z + half.z),
            m_aabb.minP, extents, rec, dirFilter, m_nodes);
    }
#endif

#ifndef __CUDA_ARCH__
    __host__ void forEachDTreeWrapperConst(std::function<void(const DTreeWrapper*)> func) const {
        std::vector<STreeNode> hostNodes(m_numNodes);
        if (m_numNodes > 0 && m_nodes != nullptr) { cudaMemcpy(hostNodes.data(), m_nodes, m_numNodes * sizeof(STreeNode), cudaMemcpyDeviceToHost); }
        for (const auto& node : hostNodes) if (node.isLeaf) func(&node.dTree);
    }

    __host__ void forEachDTreeWrapperConstP(std::function<void(const DTreeWrapper*, const Point3f&, const Vector3f&)> func) const {
        std::vector<STreeNode> hostNodes(m_numNodes);
        if (m_numNodes > 0 && m_nodes != nullptr) { cudaMemcpy(hostNodes.data(), m_nodes, m_numNodes * sizeof(STreeNode), cudaMemcpyDeviceToHost); }
        Vector3f size(m_aabb.maxP.x - m_aabb.minP.x, m_aabb.maxP.y - m_aabb.minP.y, m_aabb.maxP.z - m_aabb.minP.z);
        if (!hostNodes.empty()) { hostNodes[0].forEachLeaf(func, m_aabb.minP, size, hostNodes.data()); }
    }

    __host__ void forEachDTreeWrapper(std::function<void(DTreeWrapper*)> func) {
        std::vector<STreeNode> hostNodes(m_numNodes);
        if (m_numNodes > 0 && m_nodes != nullptr) { cudaMemcpy(hostNodes.data(), m_nodes, m_numNodes * sizeof(STreeNode), cudaMemcpyDeviceToHost); }
        for (auto& node : hostNodes) if (node.isLeaf) func(&node.dTree);
        cudaMemcpy(m_nodes, hostNodes.data(), m_numNodes * sizeof(STreeNode), cudaMemcpyHostToDevice);
    }

    __host__ void forEachDTreeWrapperParallel(std::function<void(DTreeWrapper*)> func) {
        std::vector<STreeNode> hostNodes(m_numNodes);
        if (m_numNodes > 0 && m_nodes != nullptr) { cudaMemcpy(hostNodes.data(), m_nodes, m_numNodes * sizeof(STreeNode), cudaMemcpyDeviceToHost); }
        const int n = static_cast<int>(hostNodes.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
        for (int i = 0; i < n; ++i) if (hostNodes[i].isLeaf) func(&hostNodes[i].dTree);
        cudaMemcpy(m_nodes, hostNodes.data(), m_numNodes * sizeof(STreeNode), cudaMemcpyHostToDevice);
    }

    __host__ void subdivide(int nodeIdx, std::vector<STreeNode>& nodes) {
        nodes.resize(nodes.size() + 2);
        if (nodes.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) return; 

        STreeNode& cur = nodes[nodeIdx];
        for (int i = 0; i < 2; ++i) {
            const uint32_t idx = static_cast<uint32_t>(nodes.size()) - 2 + i;
            cur.children[i] = idx;
            nodes[idx].axis  = (cur.axis + 1) % 3;
            nodes[idx].dTree = cur.dTree; 
            nodes[idx].dTree.setStatisticalWeightBuilding(nodes[idx].dTree.statisticalWeightBuilding() * 0.5f);
        }
        cur.isLeaf = false;
        cur.dTree.clear(); 
    }

    __host__ void subdivideAll() {
        std::vector<STreeNode> hostNodes(m_numNodes);
        if (m_numNodes > 0 && m_nodes != nullptr) { cudaMemcpy(hostNodes.data(), m_nodes, m_numNodes * sizeof(STreeNode), cudaMemcpyDeviceToHost); }
        const int n = static_cast<int>(hostNodes.size());
        for (int i = 0; i < n; ++i) { if (hostNodes[i].isLeaf) subdivide(i, hostNodes); }
        if (m_nodes) cudaFree(m_nodes);
        m_numNodes = hostNodes.size();
        cudaMalloc(&m_nodes, m_numNodes * sizeof(STreeNode));
        cudaMemcpy(m_nodes, hostNodes.data(), m_numNodes * sizeof(STreeNode), cudaMemcpyHostToDevice);
    }

    inline __host__ bool shallSplit(const STreeNode& node, size_t samplesRequired, size_t currentSize) const {
        return currentSize < static_cast<size_t>(std::numeric_limits<uint32_t>::max()) - 1
            && node.dTree.statisticalWeightBuilding() > static_cast<float>(samplesRequired);
    }

    __host__ void refine(size_t sTreeThreshold, int maxMB) {
        std::vector<STreeNode> hostNodes(m_numNodes);
        if (m_numNodes > 0 && m_nodes != nullptr) { cudaMemcpy(hostNodes.data(), m_nodes, m_numNodes * sizeof(STreeNode), cudaMemcpyDeviceToHost); }

        if (maxMB >= 0) {
            size_t approxBytes = 0;
            for (const auto& node : hostNodes) { if (node.isLeaf) approxBytes += node.dTreeWrapper()->approxMemoryFootprint(); }
            if (approxBytes / 1000000 >= static_cast<size_t>(maxMB)) return;
        }

        struct StackNode { size_t index; int depth; };
        std::stack<StackNode> stk;
        stk.push({0, 1});
        bool topologyChanged = false;

        while (!stk.empty()) {
            StackNode sn = stk.top();
            stk.pop();

            if (hostNodes[sn.index].isLeaf) {
                if (shallSplit(hostNodes[sn.index], sTreeThreshold, hostNodes.size())) {
                    subdivide(static_cast<int>(sn.index), hostNodes);
                    topologyChanged = true;
                }
            }

            if (!hostNodes[sn.index].isLeaf) {
                const STreeNode& node = hostNodes[sn.index];
                for (int i = 0; i < 2; ++i) stk.push({node.children[i], sn.depth + 1});
            }
        }

        if (topologyChanged) {
            if (m_nodes) cudaFree(m_nodes);
            m_numNodes = hostNodes.size();
            cudaMalloc(&m_nodes, m_numNodes * sizeof(STreeNode));
            cudaMemcpy(m_nodes, hostNodes.data(), m_numNodes * sizeof(STreeNode), cudaMemcpyHostToDevice);
        }
    }
#endif

    inline __host__ __device__ const AABB& aabb() const { return m_aabb; }
    inline __host__ __device__ size_t numNodes() const { return m_numNodes; }

#ifndef __CUDA_ARCH__
    __host__ size_t numLeaves() const {
        std::vector<STreeNode> hostNodes(m_numNodes);
        if (m_numNodes > 0 && m_nodes != nullptr) { cudaMemcpy(hostNodes.data(), m_nodes, m_numNodes * sizeof(STreeNode), cudaMemcpyDeviceToHost); }
        size_t n = 0;
        for (const auto& node : hostNodes) if (node.isLeaf) ++n;
        return n;
    }

    __host__ void gatherStatistics() const {
        std::vector<STreeNode> hostNodes(m_numNodes);
        if (m_numNodes > 0 && m_nodes != nullptr) { cudaMemcpy(hostNodes.data(), m_nodes, m_numNodes * sizeof(STreeNode), cudaMemcpyDeviceToHost); }

        int   maxDepth = 0, minDepth = std::numeric_limits<int>::max();
        float avgDepth = 0.f;
        float maxRad = 0.f, minRad = std::numeric_limits<float>::max(), avgRad = 0.f;
        size_t maxN = 0, minN = std::numeric_limits<size_t>::max();
        float  avgN = 0.f;
        float  maxW = 0.f, minW = std::numeric_limits<float>::max(), avgW = 0.f;
        int    cnt = 0, cntN = 0;
        size_t leavesCount = 0;

        for (const auto& node : hostNodes) {
            if (!node.isLeaf) continue;
            ++leavesCount;
            const DTreeWrapper* dw = node.dTreeWrapper();

            const int d = dw->depth();
            maxDepth = std::max(maxDepth, d);
            minDepth = std::min(minDepth, d);
            avgDepth += d;

            const float r = dw->meanRadiance();
            maxRad = std::max(maxRad, r);
            minRad = std::min(minRad, r);
            avgRad += r;

            if (dw->numNodes() > 1) {
                const size_t nn = dw->numNodes();
                maxN = std::max(maxN, nn);
                minN = std::min(minN, nn);
                avgN += static_cast<float>(nn);
                ++cntN;
            }

            const float w = dw->statisticalWeight();
            maxW = std::max(maxW, w);
            minW = std::min(minW, w);
            avgW += w;
            ++cnt;
        }

        if (cnt > 0) {
            avgDepth /= cnt;
            avgRad   /= cnt;
            avgW     /= cnt;
            if (cntN > 0) avgN /= cntN;
        }

        printf(
            "S-Tree statistics (%zu leaves):\n"
            "  D-Tree depth   = [%d, %.2f, %d]\n"
            "  Mean radiance  = [%.4f, %.4f, %.4f]\n"
            "  D-Tree nodes   = [%zu, %.1f, %zu]\n"
            "  Stat. weight   = [%.2f, %.2f, %.2f]\n",
            leavesCount, minDepth, avgDepth, maxDepth, minRad, avgRad, maxRad,
            (cntN > 0 ? minN : 0), avgN, (cntN > 0 ? maxN : 0), (cnt > 0 ? minW : 0.f), avgW, maxW);
    }
#endif
};

} // namespace futaba