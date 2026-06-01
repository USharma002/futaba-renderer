#pragma once

#include <cstdint>
#include <vector>
#include <stack>
#include <cmath>
#include <algorithm>
#include <limits>

#include "types.cuh"
#include "common.cuh"
#include "sampler.cuh"

namespace futaba {

enum class EDirectionalFilter {
    ENearest,
    EBox,
};

enum class EBsdfSamplingFractionLoss {
    ENone,
    EKL,
    EVariance,
};

// Safe cross-platform device math primitives to eliminate NVCC cross-pollution
inline __host__ __device__ float safe_min(float a, float b) { return (a < b) ? a : b; }
inline __host__ __device__ float safe_max(float a, float b) { return (a > b) ? a : b; }
inline __host__ __device__ bool safe_isfinite(float x) {
#if defined(__CUDA_ARCH__)
    return !isnan(x) && !isinf(x);
#else
    return std::isfinite(x);
#endif
}

inline __host__ __device__ void atomic_add_float(float* address, float val) {
#if defined(__CUDA_ARCH__)
    atomicAdd(address, val);
#else
    *address += val;
#endif
}

inline __host__ __device__ float logistic(float x) {
#if defined(__CUDA_ARCH__)
    return 1.f / (1.f + __expf(-x));
#else
    return 1.f / (1.f + std::exp(-x));
#endif
}

struct QuadTreeNode {
    float m_sum[4];          
    uint32_t m_children[4];  

    inline __host__ __device__ QuadTreeNode() {
        for (int i = 0; i < 4; ++i) {
            m_sum[i] = 0.0f;
            m_children[i] = 0; 
        }
    }

    inline __host__ __device__ void setSum(int index, float val) {
        m_sum[index] = val;
    }

    inline __host__ __device__ void setSum(float val) {
        for (int i = 0; i < 4; ++i) {
            m_sum[i] = val;
        }
    }

    inline __host__ __device__ float sum(int index) const {
        return m_sum[index];
    }

    inline __host__ __device__ QuadTreeNode(const QuadTreeNode& arg) {
        copyFrom(arg);
    }

    inline __host__ __device__ void copyFrom(const QuadTreeNode& arg) {
        for (int i = 0; i < 4; ++i) {
            m_sum[i] = arg.m_sum[i];
            m_children[i] = arg.m_children[i];
        }
    }

    inline __host__ __device__ QuadTreeNode& operator=(const QuadTreeNode& arg) {
        copyFrom(arg);
        return *this;
    }

#ifndef __CUDA_ARCH__
    inline __host__ __device__ float pdf(Point2f& p, const QuadTreeNode* nodes) const {
        p.x = safe_max(0.0f, safe_min(p.x, 1.0f));
        p.y = safe_max(0.0f, safe_min(p.y, 1.0f));

        const int index = childIndex(p);
        if (!(sum(index) > 0.0f)) {
            return 0.0f;
        }

        const float factor = 4.0f * sum(index) / (sum(0) + sum(1) + sum(2) + sum(3));
        if (isLeaf(index)) {
            return factor;
        } else {
            return factor * nodes[child(index)].pdf(p, nodes);
        }
    }

    inline __host__ __device__ float eval(Point2f& p, const QuadTreeNode* nodes) const {
        p.x = safe_max(0.0f, safe_min(p.x, 1.0f));
        p.y = safe_max(0.0f, safe_min(p.y, 1.0f));

        const int index = childIndex(p);
        if (isLeaf(index)) {
            return 4.0f * sum(index);
        } else {
            return 4.0f * nodes[child(index)].eval(p, nodes);
        }
    }

    template <typename SamplerType>
    inline __host__ __device__ Point2f sample(SamplerType& sampler, const QuadTreeNode* nodes) const {
        int index = 0;

        float topLeft = sum(0);                     
        float topRight = sum(1);                    
        float partial = topLeft + sum(2);           
        float total = partial + topRight + sum(3);  

        if (!(total > 0.0f)) {
            return sampler.next2D();
        }

        float boundary = partial / total;
        Point2f origin = Point2f{0.0f, 0.0f};
        float sampleVal = sampler.next1D();

        if (sampleVal < boundary) {
            sampleVal /= boundary; 
            boundary = topLeft / partial; 
        } else {
            partial = total - partial;
            origin.x = 0.5f;
            sampleVal = (sampleVal - boundary) / (1.0f - boundary);
            boundary = topRight / partial;
            index |= 1 << 0;
        }

        if (sampleVal < boundary) {
            sampleVal /= boundary;
        } else {
            origin.y = 0.5f;
            sampleVal = (sampleVal - boundary) / (1.0f - boundary);
            index |= 1 << 1;
        }

        if (isLeaf(index)) {
            return origin + 0.5f * sampler.next2D();
        } else {
            return origin + 0.5f * nodes[child(index)].sample(sampler, nodes);
        }
    }

    inline __host__ __device__ void record(Point2f& p, float irradiance, QuadTreeNode* nodes) {
        p.x = safe_max(0.0f, safe_min(p.x, 1.0f));
        p.y = safe_max(0.0f, safe_min(p.y, 1.0f));

        int index = childIndex(p);

        if (isLeaf(index)) {
            atomic_add_float(&m_sum[index], irradiance);
        } else {
            nodes[child(index)].record(p, irradiance, nodes);
        }
    }

    inline __host__ __device__ float computeOverlappingArea(const Point2f &min1, const Point2f &max1, const Point2f &min2, const Point2f &max2) const {
        float xOverlap = safe_max(0.0f, safe_min(max1.x, max2.x) - safe_max(min1.x, min2.x));
        float yOverlap = safe_max(0.0f, safe_min(max1.y, max2.y) - safe_max(min1.y, min2.y));
        return xOverlap * yOverlap; 
    }

    inline __host__ __device__ void record(const Point2f& origin, float size, Point2f nodeOrigin, float nodeSize, float value, QuadTreeNode* nodes) {
        float childSize = nodeSize / 2.0f;
        for (int i = 0; i < 4; ++i) {
            Point2f childOrigin = nodeOrigin;
            if (i & 1) { childOrigin.x += childSize; }
            if (i & 2) { childOrigin.y += childSize; }

            float w = computeOverlappingArea(origin, Point2f(size) + origin, childOrigin, childOrigin + Point2f(childSize));
            if (w > 0.0f) {
                if (isLeaf(i)) {
                    atomic_add_float(&m_sum[i], value * w);
                } else {
                    nodes[child(i)].record(origin, size, childOrigin, childSize, value, nodes); 
                }
            }
        }
    }

    inline __host__ __device__ int depthAt(Point2f p, const QuadTreeNode* nodes) const {
        const int index = childIndex(p);
        if (isLeaf(index)) {
            return 1;
        } else {
            return 1 + nodes[child(index)].depthAt(p, nodes);
        }
    }
#endif

    inline __host__ __device__ uint32_t child(int index) const {
        return m_children[index];
    }

    inline __host__ __device__ void setChild(int index, uint32_t val) {
        m_children[index] = val; 
    }

    inline __host__ __device__ int childIndex(Point2f &p) const {
        int res = 0;
        if (p.x < 0.5f) {
            p.x *= 2.0f;
        } else {
            p.x = (p.x - 0.5f) * 2.0f;
            res |= 1;
        }

        if (p.y < 0.5f) {
            p.y *= 2.0f;
        } else {
            p.y = (p.y - 0.5f) * 2.0f;
            res |= 2;
        }
        return res;
    }

    inline __host__ __device__ bool isLeaf(int index) const {
        return m_children[index] == 0;
    }

#ifndef __CUDA_ARCH__
    inline __host__ void build(QuadTreeNode* nodes, int distributionMode = 0, float parentSize = 1.f) {
        float childSize = parentSize / 4.f;
        for (int i = 0; i < 4; ++i) {
            if (isLeaf(i)) {
                if (distributionMode == 2 || distributionMode == 3) { // L2 / Second Moment / Full L2
                    setSum(i, std::sqrt(sum(i) * childSize));
                }
                continue;
            }
            QuadTreeNode& c = nodes[child(i)];
            c.build(nodes, distributionMode, childSize);

            float calculatedSum = 0.0f;
            for (int j = 0; j < 4; ++j) {
                calculatedSum += c.sum(j);
            }
            setSum(i, calculatedSum);
        }
    }  
#endif
};

struct DTreeRecord {
    Vector3f d;
    float radiance, product;
    float woPdf, bsdfPdf, dTreePdf;
    float statisticalWeight;
    bool isDelta;
};

struct DTree {
    QuadTreeNode* m_nodes; 
    size_t m_numNodes;

    float m_sum;
    float m_statisticalWeight;
    int m_maxDepth;

#ifndef __CUDA_ARCH__
    __host__ void initialize() {
        m_sum = 0.0f;
        m_statisticalWeight = 0.0f;
        m_maxDepth = 1;
        m_numNodes = 1;

        std::vector<QuadTreeNode> rootNode(1);
        cudaMalloc(&m_nodes, sizeof(QuadTreeNode));
        cudaMemcpy(m_nodes, rootNode.data(), sizeof(QuadTreeNode), cudaMemcpyHostToDevice);
    }

    __host__ void clear() {
        if (m_nodes) {
            cudaFree(m_nodes);
            m_nodes = nullptr;
        }
        m_numNodes = 0;
        m_sum = 0.0f;
        m_statisticalWeight = 0.0f;
        m_maxDepth = 0;
    }

    __host__ DTree() : m_nodes(nullptr), m_numNodes(0), m_sum(0.0f), m_statisticalWeight(0.0f), m_maxDepth(0) {
        initialize();
    }

    __host__ ~DTree() {}

    __host__ DTree& operator=(const DTree& other) {
        if (this != &other) {
            clear();
            m_numNodes = other.m_numNodes;
            m_maxDepth = other.m_maxDepth;
            m_sum = other.m_sum;
            m_statisticalWeight = other.m_statisticalWeight;

            if (m_numNodes > 0 && other.m_nodes != nullptr) {
                cudaMalloc(&m_nodes, m_numNodes * sizeof(QuadTreeNode));
                cudaMemcpy(m_nodes, other.m_nodes, m_numNodes * sizeof(QuadTreeNode), cudaMemcpyDeviceToDevice);
            }
        }
        return *this;
    }

    __host__ DTree(const DTree& other) : m_nodes(nullptr), m_numNodes(0), m_sum(0.f), m_statisticalWeight(0.f), m_maxDepth(0) {
        *this = other;
    }
#endif

    inline __host__ __device__ const QuadTreeNode& node(size_t i) const { return m_nodes[i]; }
    inline __host__ __device__ QuadTreeNode& node(size_t i) { return m_nodes[i]; }

    inline __host__ __device__ float mean() const {
        if (m_statisticalWeight == 0.0f) {
            return 0.0f;
        }
        const float factor = 1.0f / (M_PI * 4.0f * m_statisticalWeight);
        return factor * m_sum; 
    }

    inline __host__ __device__ float eval(Point2f p) const {
        p.x = safe_max(0.0f, safe_min(p.x, 1.0f));
        p.y = safe_max(0.0f, safe_min(p.y, 1.0f));

        const QuadTreeNode* curr = &m_nodes[0];
        float factor = 1.f;

        while (true) {
            const int index = curr->childIndex(p);
            if (curr->isLeaf(index)) {
                return factor * 4.0f * curr->sum(index);
            } else {
                factor *= 4.0f;
                curr = &m_nodes[curr->child(index)];
            }
        }
    }

    inline __host__ __device__ float pdf(Point2f p) const {
        if (!(mean() > 0.0f)) {
            return 1.0f / (4.0f * M_PI); 
        }
        
        p.x = safe_max(0.0f, safe_min(p.x, 1.0f));
        p.y = safe_max(0.0f, safe_min(p.y, 1.0f));

        const QuadTreeNode* curr = &m_nodes[0];
        float factor = 1.f;

        while (true) {
            const int index = curr->childIndex(p);
            float sumIndex = curr->sum(index);
            if (!(sumIndex > 0.0f)) {
                return 0.f;
            }

            float total = curr->sum(0) + curr->sum(1) + curr->sum(2) + curr->sum(3);
            factor *= 4.0f * sumIndex / total;

            if (curr->isLeaf(index)) {
                return factor / (4.0f * M_PI);
            } else {
                curr = &m_nodes[curr->child(index)];
            }
        }
    }
    
    template <typename SamplerType>
    inline __host__ __device__ Point2f sample(SamplerType &sampler) const {
        if (!(mean() > 0.0f)) {
            return sampler.next2D();
        }

        const QuadTreeNode* curr = &m_nodes[0];
        Point2f origin = Point2f(0.f, 0.f);
        float scale = 1.f;

        while (true) {
            int index = 0;
            float topLeft = curr->sum(0);                     
            float topRight = curr->sum(1);                    
            float partial = topLeft + curr->sum(2);           
            float total = partial + topRight + curr->sum(3);  

            if (!(total > 0.0f)) {
                Point2f nextSample = sampler.next2D();
                Point2f res = origin + scale * nextSample;
                res.x = safe_max(0.0f, safe_min(res.x, 1.0f));
                res.y = safe_max(0.0f, safe_min(res.y, 1.0f));
                return res;
            }

            float boundary = partial / total;
            Point2f childOffset = Point2f(0.0f, 0.0f);
            float sampleVal = sampler.next1D();

            if (sampleVal < boundary) {
                sampleVal /= boundary; 
                boundary = topLeft / partial; 
            } else {
                partial = total - partial;
                childOffset.x = 0.5f;
                sampleVal = (sampleVal - boundary) / (1.0f - boundary);
                boundary = topRight / partial;
                index |= 1 << 0;
            }

            if (sampleVal < boundary) {
                sampleVal /= boundary;
            } else {
                childOffset.y = 0.5f;
                sampleVal = (sampleVal - boundary) / (1.0f - boundary);
                index |= 1 << 1;
            }

            if (curr->isLeaf(index)) {
                Point2f nextSample = sampler.next2D();
                Point2f res = origin + scale * (childOffset + 0.5f * nextSample);
                res.x = safe_max(0.0f, safe_min(res.x, 1.0f));
                res.y = safe_max(0.0f, safe_min(res.y, 1.0f));
                return res;
            } else {
                origin = origin + scale * childOffset;
                scale *= 0.5f;
                curr = &m_nodes[curr->child(index)];
            }
        }
    }

    inline __host__ __device__ int depth() const { return m_maxDepth; }
    inline __host__ __device__ int depthAt(Point2f p) const {
        p.x = safe_max(0.0f, safe_min(p.x, 1.0f));
        p.y = safe_max(0.0f, safe_min(p.y, 1.0f));
        const QuadTreeNode* curr = &m_nodes[0];
        int d = 1;
        while (true) {
            const int index = curr->childIndex(p);
            if (curr->isLeaf(index)) {
                return d;
            } else {
                d++;
                curr = &m_nodes[curr->child(index)];
            }
        }
    }
    inline __host__ __device__ size_t numNodes() const { return m_numNodes; }
    inline __host__ __device__ float statisticalWeight() const { return m_statisticalWeight; }
    inline __host__ __device__ void setStatisticalWeight(float val) { m_statisticalWeight = val; }

#ifndef __CUDA_ARCH__
    inline __host__ __device__ void recordIrradiance(Point2f p, float irradiance, float statisticalWeight, EDirectionalFilter directionalFilter) {
        if (safe_isfinite(statisticalWeight) && statisticalWeight > 0.0f) {
            atomic_add_float(&m_statisticalWeight, statisticalWeight);

            if (safe_isfinite(irradiance) && irradiance > 0.0f) {
                atomic_add_float(&m_sum, irradiance * statisticalWeight);

                if (directionalFilter == EDirectionalFilter::ENearest) {
                    m_nodes[0].record(p, irradiance * statisticalWeight, m_nodes);
                } else {
                    int currentDepth = depthAt(p);
                    float size = std::pow(0.5f, currentDepth);

                    Point2f origin = p;
                    origin.x -= size / 2.0f;
                    origin.y -= size / 2.0f;
                    m_nodes[0].record(origin, size, Point2f(0.0f), 1.0f, irradiance * statisticalWeight / (size * size), m_nodes);
                }
            }
        }
    }
#endif

#ifndef __CUDA_ARCH__
    struct StackNode {
        size_t nodeIndex;
        size_t otherNodeIndex;
        const std::vector<QuadTreeNode>* otherNodes;
        int depth;
    };

    __host__ void reset(const DTree& previousDTree, int newMaxDepth, float subdivisionThreshold) {
        m_sum = 0.0f;
        m_statisticalWeight = 0.0f;
        m_maxDepth = 0;

        std::vector<QuadTreeNode> hostNodes;
        hostNodes.emplace_back();       

        std::vector<QuadTreeNode> oldHostNodes(previousDTree.m_numNodes);
        if (previousDTree.m_numNodes > 0 && previousDTree.m_nodes != nullptr) {
            cudaMemcpy(oldHostNodes.data(), previousDTree.m_nodes, previousDTree.m_numNodes * sizeof(QuadTreeNode), cudaMemcpyDeviceToHost);
        }

        std::stack<StackNode> nodeIndices;
        nodeIndices.push(StackNode{0, 0, &oldHostNodes, 1});

        const float total = previousDTree.m_sum;
        
        while (!nodeIndices.empty()) {
            StackNode sNode = nodeIndices.top();
            nodeIndices.pop();

            m_maxDepth = std::max(m_maxDepth, sNode.depth);

            for (int i = 0; i < 4; ++i) {
                const QuadTreeNode& otherNode = (*sNode.otherNodes)[sNode.otherNodeIndex];
                const float fraction = total > 0.0f ? (otherNode.sum(i) / total) : std::pow(0.25f, sNode.depth);

                if (sNode.depth < newMaxDepth && fraction > subdivisionThreshold) {
                    uint32_t targetChildIndex = static_cast<uint32_t>(hostNodes.size());

                    if (!otherNode.isLeaf(i)) {
                        nodeIndices.push(StackNode{targetChildIndex, otherNode.child(i), &oldHostNodes, sNode.depth + 1});
                    } else {
                        nodeIndices.push(StackNode{targetChildIndex, targetChildIndex, &hostNodes, sNode.depth + 1});
                    }

                    hostNodes[sNode.nodeIndex].setChild(i, targetChildIndex);
                    hostNodes.emplace_back();
                    hostNodes.back().setSum((*sNode.otherNodes)[sNode.otherNodeIndex].sum(i) / 4.f);

                    if (hostNodes.size() > std::numeric_limits<uint32_t>::max()) {
                        nodeIndices = std::stack<StackNode>();
                        break;
                    }
                }
            }
        }

        for (auto& node : hostNodes) {
            node.setSum(0.0f);
        }

        if (m_nodes) { cudaFree(m_nodes); }
        
        m_numNodes = hostNodes.size();
        cudaMalloc(&m_nodes, m_numNodes * sizeof(QuadTreeNode));
        cudaMemcpy(m_nodes, hostNodes.data(), m_numNodes * sizeof(QuadTreeNode), cudaMemcpyHostToDevice);
    }

    size_t approxMemoryFootprint() const { return m_numNodes * sizeof(QuadTreeNode) + sizeof(*this); }

    __host__ void build(int distributionMode = 0) {
        if (m_numNodes == 0 || !m_nodes) return;

        std::vector<QuadTreeNode> hostNodes(m_numNodes);
        cudaMemcpy(hostNodes.data(), m_nodes, m_numNodes * sizeof(QuadTreeNode), cudaMemcpyDeviceToHost);

        hostNodes[0].build(hostNodes.data(), distributionMode, 1.f);

        float overallSum = 0.0f;
        for (int i = 0; i < 4; ++i) { overallSum += hostNodes[0].sum(i); }
        m_sum = overallSum;

        cudaMemcpy(m_nodes, hostNodes.data(), m_numNodes * sizeof(QuadTreeNode), cudaMemcpyHostToDevice);
    }
#endif
};

struct DTreeWrapper {
    DTree building;   
    DTree sampling;   

#ifndef __CUDA_ARCH__
    inline __host__ void clear() {
        building.clear();
        sampling.clear();
    }

    inline DTreeWrapper& operator=(const DTreeWrapper& other) {
        if (this != &other) {
            building = other.building;
            sampling = other.sampling;
        }
        return *this;
    }

    inline DTreeWrapper(const DTreeWrapper& other) {
        *this = other;
    }
    DTreeWrapper() = default;
#endif

    inline static __host__ __device__ Vector3f canonicalToDir(Point2f p) {
        const float cosTheta = 2.f * p.x - 1.f;
        const float phi      = 2.f * M_PI * p.y;
        const float sinTheta = std::sqrt(safe_max(0.f, 1.f - cosTheta * cosTheta));

        float sinPhi, cosPhi;
#if defined(__CUDA_ARCH__)
        sincosf(phi, &sinPhi, &cosPhi);
#else
        sinPhi = std::sin(phi);
        cosPhi = std::cos(phi);
#endif
        return Vector3f(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
    }

    inline static __host__ __device__ Point2f dirToCanonical(const Vector3f& d) {
        if (!safe_isfinite(d.x) || !safe_isfinite(d.y) || !safe_isfinite(d.z)) {
            return Point2f(0.f, 0.f);
        }

        const float cosTheta = safe_min(safe_max(d.z, -1.0f), 1.0f);
        float phi = std::atan2(d.y, d.x);
        while (phi < 0.0f) phi += 2.0f * M_PI;

        return Point2f((cosTheta + 1.f) * 0.5f, phi / (2.f * M_PI));
    }

    template <typename SamplerType>
    inline __host__ __device__ Vector3f sample(SamplerType& sampler) const {
        return canonicalToDir(sampling.sample(sampler));
    }

    inline __host__ __device__ float pdf(const Vector3f& dir) const { return sampling.pdf(dirToCanonical(dir)); }
    inline __host__ __device__ float diff(const DTreeWrapper& other) const { return 0.0f; }

#ifndef __CUDA_ARCH__
    inline __host__ __device__ void record(const DTreeRecord& rec, EDirectionalFilter dirFilter) {
        if (rec.isDelta) return;  
        if (!(rec.woPdf > 0.f))  return;

        const float irradiance = rec.radiance / rec.woPdf;
        building.recordIrradiance(dirToCanonical(rec.d), irradiance, rec.statisticalWeight, dirFilter);
    }
#endif

#ifndef __CUDA_ARCH__
    __host__ void build(int distributionMode = 0) {
        building.build(distributionMode);
        sampling.clear();
        sampling = building; 
    }

    __host__ void reset(int maxDepth, float subdivisionThreshold) { building.reset(sampling, maxDepth, subdivisionThreshold); }
#endif
    inline __host__ __device__ int depth() const { return sampling.depth(); }
    inline __host__ __device__ size_t numNodes() const { return sampling.numNodes(); }
    inline __host__ __device__ float meanRadiance() const { return sampling.mean(); }
    inline __host__ __device__ float statisticalWeight() const { return sampling.statisticalWeight(); }
    inline __host__ __device__ float statisticalWeightBuilding() const { return building.statisticalWeight(); }
    inline __host__ __device__ void setStatisticalWeightBuilding(float statisticalWeight) { building.setStatisticalWeight(statisticalWeight); }
#ifndef __CUDA_ARCH__
    size_t approxMemoryFootprint() const { return building.approxMemoryFootprint() + sampling.approxMemoryFootprint(); }
#endif
    inline __host__ __device__ float bsdfSamplingFraction(float variable) const { return logistic(variable); }
};

} // namespace futaba