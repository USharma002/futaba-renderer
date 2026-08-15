#pragma once

#include "types.cuh"
#include "common.cuh"
#include "warp.cuh"
#include "frame.cuh"
#include "bsdf_sample.cuh"
#include "bsdf.cuh"
#include "surface_interaction.cuh"
#include "ppg/stree.h"

namespace futaba {

// Destructor-free view over the device S-tree. STree itself owns device memory
// (and frees it in its destructor), so we never copy an STree onto the GPU --
// we hand the integrator the raw node pointer + bounds and wrap them here.
struct STreeView {
    STreeNode* nodes;
    AABB       aabb;

    HD DTreeWrapper* dTreeWrapper(Point3f p) const {
        Vector3f size = aabb.extent();
        p = Point3f((p.x - aabb.min.x) / size.x,
                    (p.y - aabb.min.y) / size.y,
                    (p.z - aabb.min.z) / size.z);
        return nodes[0].dTreeWrapper(p, size, nodes);
    }
};

// PPG (SD-Tree) Model
struct PPGDistribution {
    DTreeWrapper* dw = nullptr;

    HD PPGDistribution() : dw(nullptr) {}
    HD PPGDistribution(const SurfaceInteraction& si, const GuidingParams& params) {
        if (params.sTreeNodes != nullptr) {
            STreeView view{params.sTreeNodes, params.sTreeAABB};
            DTreeWrapper* d = view.dTreeWrapper(si.p);
            dw = (d && d->meanRadiance() > 0.f) ? d : nullptr;
        }
    }

    HD bool is_valid() const { return dw != nullptr; }

    HD float pdf(const Vector3f& wo_world) const {
        return dw ? dw->pdf(wo_world) : 0.f;
    }

    template <typename SamplerType>
    HD Vector3f sample(SamplerType& sampler, float& pdf_out) const {
        if (!dw) {
            pdf_out = 0.f;
            return Vector3f(0.f);
        }
        Vector3f wo_world = dw->sample(sampler);
        pdf_out = dw->pdf(wo_world);
        return wo_world;
    }
};

// Cosine-Weighted Hemisphere Fallback
struct CosineDistribution {
    HD static float pdf(const Vector3f& wo_local) {
        if (Frame::cos_theta(wo_local) <= 0.f) return 0.f;
        return Warp::squareToCosineHemispherePdf(wo_local);
    }

    template <typename SamplerType>
    HD static Vector3f sample(SamplerType& sampler, float& pdf_out) {
        Vector3f wo_local = Warp::squareToCosineHemisphere(sampler.next2D());
        pdf_out = Warp::squareToCosineHemispherePdf(wo_local);
        return wo_local;
    }
};

// Unified Guiding Distribution Wrapper
struct GuidingDistribution {
    Frame           frame;
    GuidingType     type = GuidingType::None;
    PPGDistribution ppg;

    HD GuidingDistribution() : frame(), type(GuidingType::None) {}

    HD GuidingDistribution(const SurfaceInteraction& si, const GuidingParams& params)
        : frame(si.frame), type(params.active ? params.type : GuidingType::None)
    {
        if (!params.active) return;
        if (type == GuidingType::PPG) {
            ppg = PPGDistribution(si, params);
        }
    }

    HD bool is_valid() const {
        if (type == GuidingType::PPG) {
            return ppg.is_valid();
        }
        return false;
    }

    // Solid-angle PDF of scattering into local direction wo_local
    HD float pdf(const Vector3f& wo_local) const {
        if (Frame::cos_theta(wo_local) <= 0.f) return 0.f;
        float p = 0.f;
        if (type == GuidingType::PPG && ppg.is_valid()) {
            p = ppg.pdf(frame.to_world(wo_local));
        }
        if (p <= 0.f || isnan(p) || isinf(p)) {
            return CosineDistribution::pdf(wo_local);
        }
        return p;
    }

    // Combined BSDF / guiding PDF used for MIS weighting
    HD float mixed_pdf(const Vector3f& wo_local, float bsdf_pdf, float alpha) const {
        if (type == GuidingType::None) return bsdf_pdf;
        return alpha * bsdf_pdf + (1.f - alpha) * pdf(wo_local);
    }

    // Sample a direction from the guiding distribution into local space
    template <typename SamplerType>
    HD void sample(BSDFSample& bs, SamplerType& sampler) const {
        if (type == GuidingType::PPG && ppg.is_valid()) {
            Vector3f wo_world = ppg.sample(sampler, bs.pdf);
            bs.wo = frame.to_local(wo_world);
            if (bs.pdf > 0.f && !isnan(bs.pdf) && !isinf(bs.pdf) && Frame::cos_theta(bs.wo) > 0.f) {
                return;
            }
        }
        bs.wo = CosineDistribution::sample(sampler, bs.pdf);
    }
};

// Guiding Dispatcher
struct Guiding {
    // Check whether guiding should be applied to a given material interaction
    HD static bool is_guidable(const Material& mat, const GuidingParams& params) {
        return params.active && params.type != GuidingType::None &&
               !BSDF::is_delta(mat.type) && mat.type != BSDF_ID_NULL;
    }

    // Construct the active guiding distribution for a surface interaction
    HD static GuidingDistribution get_distribution(const SurfaceInteraction& si, const GuidingParams& params) {
        return GuidingDistribution(si, params);
    }

    // One-sample MIS between BSDF sampling and guided sampling
    template <typename SamplerType>
    HD static Color3f sample_guided(const GuidingDistribution& guide,
                                    const Material& mat,
                                    const SurfaceInteraction& si,
                                    BSDFSample& bs,
                                    SamplerType& sampler,
                                    float alpha)
    {
        if (!guide.is_valid() && guide.type == GuidingType::None)
            return BSDF::sample(mat, si, bs, sampler.next2D());

        if (sampler.next1D() < alpha) {
            // BSDF Strategy
            Color3f f_weight = BSDF::sample(mat, si, bs, sampler.next2D());
            if (!bs.is_valid()) return Color3f(0.f);
            float mixed = alpha * bs.pdf + (1.f - alpha) * guide.pdf(bs.wo);
            if (mixed <= 0.f) { bs.pdf = 0.f; return Color3f(0.f); }
            Color3f w = f_weight * (bs.pdf / mixed);
            bs.pdf = mixed;
            return w;
        } else {
            // Guided Strategy
            guide.sample(bs, sampler);
            bs.sampled_type = BSDF_ID_DIFFUSE;
            bs.eta = 1.f;
            if (!bs.is_valid()) return Color3f(0.f);
            Color3f f_bsdf;
            float bsdf_pdf = 0.f;
            BSDF::eval_pdf(mat, si, bs.wo, f_bsdf, bsdf_pdf);
            float mixed = alpha * bsdf_pdf + (1.f - alpha) * bs.pdf;
            if (mixed <= 0.f) { bs.pdf = 0.f; return Color3f(0.f); }
            Color3f w = f_bsdf * (Frame::abs_cos_theta(bs.wo) / mixed);
            bs.pdf = mixed;
            return w;
        }
    }
};

} // namespace futaba
