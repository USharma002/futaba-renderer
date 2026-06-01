#pragma once

#include "types.cuh"
#include "common.cuh"
#include "warp.cuh"
#include "frame.cuh"
#include "bsdf_sample.cuh"
#include "guiding.h"
#include "surface_interaction.cuh"
#include "ppg/stree.h" 

namespace futaba {

// State-free LCG fractional pseudo-sampler maps s2 fields into state jumps
struct PseudoSampler {
    float x, y;
    int state;

    __host__ __device__ PseudoSampler(Point2f s) : x(s.x), y(s.y), state(0) {}

    __host__ __device__ float safe_fracf(float v) const {
        return v - floorf(v);
    }

    __host__ __device__ float next1D() {
        state++;
        float seed = x * 12.9898f + y * 78.233f + static_cast<float>(state) * 45.164f;
        return safe_fracf(sinf(seed) * 43758.5453f);
    }

    __host__ __device__ Point2f next2D() {
        return Point2f(next1D(), next1D());
    }
};

// Lightweight, destructor-free reader block processes layout navigation inside device spaces safely
struct STreeView {
    STreeNode* m_nodes;
    AABB       m_aabb;

    __host__ __device__ inline DTreeWrapper* dTreeWrapper(Point3f p, Vector3f& size) const {
        size = Vector3f(m_aabb.maxP.x - m_aabb.minP.x,
                        m_aabb.maxP.y - m_aabb.minP.y,
                        m_aabb.maxP.z - m_aabb.minP.z);
        p.x = (p.x - m_aabb.minP.x) / size.x;
        p.y = (p.y - m_aabb.minP.y) / size.y;
        p.z = (p.z - m_aabb.minP.z) / size.z;
        return m_nodes[0].dTreeWrapper(p, size, m_nodes);
    }

    __host__ __device__ inline DTreeWrapper* dTreeWrapper(Point3f p) const {
        Vector3f size;
        return dTreeWrapper(p, size);
    }
};

struct GuidingDistribution {
    int mode;
    Point3f p;
    Normal3f n;
    Frame frame;
    
    STreeNode* sTreeNodes; 
    AABB       sTreeAABB;

    __host__ __device__ inline GuidingDistribution(int m, const SurfaceIntersection& si, STreeNode* nodes, const AABB& aabb)
        : mode(m), p(si.p), n(si.n), frame(si.frame), sTreeNodes(nodes), sTreeAABB(aabb) {}

    __host__ __device__ inline Color3f eval(const BSDFSample& bs) const {
        if (mode == PATH_GUIDING_NONE || !sTreeNodes) return Color3f(0.f);
        
        if (mode == PATH_GUIDING_PPG) {
            STreeView view{sTreeNodes, sTreeAABB};

            Vector3f voxelSize;
            DTreeWrapper* dw = view.dTreeWrapper(p, voxelSize); 
            if (!dw || dw->meanRadiance() <= 0.f) return Color3f(0.f);

            // Convert incoming local direction to world space for tree lookup
            Vector3f wo_world = frame.to_world(bs.wo);
            Point2f canonicalDir = DTreeWrapper::dirToCanonical(wo_world);
            float value = dw->sampling.node(0).eval(canonicalDir, dw->sampling.m_nodes); 
            return Color3f(value / (4.f * 3.14159265358979323846f));
        }
        return Color3f(0.f);
    }

    __host__ __device__ inline float pdf(const BSDFSample& bs) const {
        if (mode == PATH_GUIDING_NONE || !sTreeNodes) return 0.f;
        if (Frame::cos_theta(bs.wo) <= 0.f) return 0.f;

        if (mode == PATH_GUIDING_PPG) {
            STreeView view{sTreeNodes, sTreeAABB};

            DTreeWrapper* dw = view.dTreeWrapper(p); 
            if (!dw || dw->meanRadiance() <= 0.f) {
                return Warp::squareToCosineHemispherePdf(bs.wo);
            }
            // Convert local ray configuration to world space coordinates
            Vector3f wo_world = frame.to_world(bs.wo);
            return dw->pdf(wo_world); 
        }
        return Warp::squareToCosineHemispherePdf(bs.wo);
    }

    __host__ __device__ inline Color3f sample(BSDFSample& bs, const Point2f& s2) const {
        if (mode == PATH_GUIDING_NONE || !sTreeNodes) {
            bs.pdf = 0.f; return Color3f(0.f);
        }

        if (mode == PATH_GUIDING_PPG) {
            STreeView view{sTreeNodes, sTreeAABB};

            DTreeWrapper* dw = view.dTreeWrapper(p); 
            
            if (!dw || dw->meanRadiance() <= 0.f) {
                bs.wo = Warp::squareToCosineHemisphere(s2);
                bs.pdf = Warp::squareToCosineHemispherePdf(bs.wo);
                bs.weight = Color3f(1.f);
                bs.sampled_type = BSDF_ID_DIFFUSE;
                return bs.weight;
            }

            PseudoSampler pseudoSampler(s2);
            Point2f canonicalSample = dw->sampling.sample(pseudoSampler); 
            Vector3f wo_world = DTreeWrapper::canonicalToDir(canonicalSample); 
            
            // Map the sampled world space vector back into local space for the material shaders
            bs.wo = frame.to_local(wo_world);
            bs.pdf = dw->pdf(wo_world); 
            
            if (bs.pdf <= 0.f || Frame::cos_theta(bs.wo) <= 0.f) {
                bs.wo = Warp::squareToCosineHemisphere(s2);
                bs.pdf = Warp::squareToCosineHemispherePdf(bs.wo);
            }

            bs.weight = Color3f(1.f);
            bs.eta = 1.f;
            bs.sampled_type = BSDF_ID_DIFFUSE;
            return bs.weight;
        }
        return Color3f(0.f);
    }
};

} // namespace futaba