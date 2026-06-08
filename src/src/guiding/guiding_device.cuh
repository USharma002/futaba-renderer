#pragma once

#include "types.cuh"
#include "common.cuh"
#include "warp.cuh"
#include "frame.cuh"
#include "bsdf_sample.cuh"
#include "bsdf.cuh"
#include "guiding.h"
#include "surface_interaction.cuh"
#include "ppg/stree.h" 

namespace futaba {

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
            float value = dw->sampling.eval(canonicalDir); 
            return Color3f(value * INV_FOURPI);
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

    template <typename SamplerType>
    __host__ __device__ inline Color3f sample(BSDFSample& bs, SamplerType& sampler) const {
        if (mode == PATH_GUIDING_NONE || !sTreeNodes) {
            bs.pdf = 0.f; return Color3f(0.f);
        }

        if (mode == PATH_GUIDING_PPG) {
            STreeView view{sTreeNodes, sTreeAABB};

            DTreeWrapper* dw = view.dTreeWrapper(p); 
            
            if (!dw || dw->meanRadiance() <= 0.f) {
                bs.wo = Warp::squareToCosineHemisphere(sampler.next2D());
                bs.pdf = Warp::squareToCosineHemispherePdf(bs.wo);
                bs.weight = Color3f(1.f);
                bs.sampled_type = BSDF_ID_DIFFUSE;
                return bs.weight;
            }

            Point2f canonicalSample = dw->sampling.sample(sampler); 
            Vector3f wo_world = DTreeWrapper::canonicalToDir(canonicalSample); 
            
            // Map the sampled world space vector back into local space for the material shaders
            bs.wo = frame.to_local(wo_world);
            bs.pdf = dw->pdf(wo_world); 
            
            if (bs.pdf <= 0.f || Frame::cos_theta(bs.wo) <= 0.f) {
                bs.pdf = 0.f;
                bs.weight = Color3f(0.f);
                return bs.weight;
            }

            bs.weight = Color3f(1.f);
            bs.eta = 1.f;
            bs.sampled_type = BSDF_ID_DIFFUSE;
            return bs.weight;
        }
        return Color3f(0.f);
    }

    __host__ __device__ inline float mixed_pdf(const Vector3f& wo_local, float bsdf_pdf, float alpha) const {
        if (mode == PATH_GUIDING_PPG && sTreeNodes != nullptr) {
            BSDFSample temp_bs;
            temp_bs.wo = wo_local;
            float dTreePdf = pdf(temp_bs);
            return alpha * bsdf_pdf + (1.f - alpha) * dTreePdf;
        }
        return bsdf_pdf;
    }

    template <typename SamplerType>
    __host__ __device__ inline Color3f sample_guided(const Material& mat,
                                                     const SurfaceIntersection& si,
                                                     BSDFSample& bs,
                                                     SamplerType& sampler,
                                                     float alpha,
                                                     float& dTreePdf) const {
        if (mode == PATH_GUIDING_PPG && sTreeNodes != nullptr && !BSDF::is_delta(mat.type) && mat.type != BSDF_ID_NULL) {
            if (sampler.next1D() < alpha) {
                // Sample BSDF
                Color3f bsdf_w = BSDF::sample(mat, si, bs, sampler.next2D());
                if (bs.is_valid()) {
                    dTreePdf = pdf(bs);
                    float mixed_pdf_val = alpha * bs.pdf + (1.f - alpha) * dTreePdf;
                    if (mixed_pdf_val > 0.f) {
                        bsdf_w = bsdf_w * (bs.pdf / mixed_pdf_val);
                        bs.pdf = mixed_pdf_val;
                        return bsdf_w;
                    } else {
                        bs.pdf = 0.f;
                    }
                }
            } else {
                // Sample DTree
                sample(bs, sampler);
                if (bs.is_valid()) {
                    Color3f f_bsdf;
                    float bsdfPdf = 0.f;
                    BSDF::eval_pdf(mat, si, bs.wo, f_bsdf, bsdfPdf);
                    
                    dTreePdf = bs.pdf; // the pdf returned from sample() is the DTree pdf
                    float mixed_pdf_val = alpha * bsdfPdf + (1.f - alpha) * dTreePdf;
                    if (mixed_pdf_val > 0.f) {
                        float cos_theta = Frame::abs_cos_theta(bs.wo);
                        Color3f bsdf_w = f_bsdf * (cos_theta / mixed_pdf_val);
                        bs.pdf = mixed_pdf_val;
                        return bsdf_w;
                    } else {
                        bs.pdf = 0.f;
                    }
                }
            }
            return Color3f(0.f);
        } else {
            dTreePdf = 0.f;
            return BSDF::sample(mat, si, bs, sampler.next2D());
        }
    }
};

} // namespace futaba
