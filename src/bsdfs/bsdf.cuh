#pragma once

#include "material.cuh"
#include "surface_interaction.cuh"
#include "bsdf_sample.cuh"
#include "diffuse.cuh"
#include "dielectric.cuh"
#include "mirror.cuh"
#include "microfacet.cuh"
#include "thindielectric.cuh"
#include "roughdielectric.cuh"
#include "roughconductor.cuh"
#include "roughplastic.cuh"

FUTABA_NAMESPACE_BEGIN

struct BSDF {
    // Check if the BSDF is Dirac-delta (no continuous PDF for MIS)
    HD static bool is_delta(BSDFType type) {
        return type == BSDF_ID_MIRROR || type == BSDF_ID_DIELECTRIC ||
               type == BSDF_ID_THINDIELECTRIC || type == BSDF_ID_NULL;
    }

    // Evaluate material albedo, modulating with diffuse texture if present
    HD static Color3f get_albedo(const Material& mat, const SurfaceInteraction& si) {
        if (mat.type == BSDF_ID_ROUGHCONDUCTOR)
            return mat.specular;
        Color3f albedo = mat.albedo;
#ifdef __CUDA_ARCH__
        if (mat.texObj != 0) {
            float4 texVal = tex2D<float4>(mat.texObj, si.uv.x, si.uv.y);
            albedo = mat.albedo * Color3f(texVal.x, texVal.y, texVal.z);
        }
#endif
        return albedo;
    }

    // Sample the BSDF in local shading coordinates
    HD static Color3f sample(const Material& mat, const SurfaceInteraction& si,
                             BSDFSample& bs, const Point2f& u) 
    {
        if (mat.type == BSDF_ID_NULL) {
            bs.wo           = -si.to_local(si.wi);
            bs.sampled_type = BSDF_ID_NULL;
            bs.pdf          = 1.0f;
            bs.eta          = 1.0f;
            bs.weight       = Color3f(1.0f);
            return bs.weight;
        }

        bs.wi         = si.to_local(si.wi);
        bs.front_face = si.front_face;
        const Color3f albedo = get_albedo(mat, si);

        switch (mat.type) {
            case BSDF_ID_MIRROR:          return Mirror(albedo).sample(bs, u);
            case BSDF_ID_DIELECTRIC:      return Dielectric(albedo, mat.intIOR, mat.extIOR).sample(bs, u);
            case BSDF_ID_THINDIELECTRIC:  return ThinDielectric(albedo, mat.intIOR, mat.extIOR).sample(bs, u);
            case BSDF_ID_MICROFACET:      return Microfacet(albedo, mat.alpha, mat.extIOR, mat.intIOR, mat.isConductor, mat.conductorEta, mat.conductorK, mat.specular).sample(bs, u);
            case BSDF_ID_ROUGHDIELECTRIC: return RoughDielectric(albedo, mat.alpha, mat.extIOR, mat.intIOR).sample(bs, u);
            case BSDF_ID_ROUGHCONDUCTOR:  return RoughConductor(albedo, mat.alpha, mat.extIOR, mat.conductorEta, mat.conductorK, mat.specular).sample(bs, u);
            case BSDF_ID_ROUGHPLASTIC:    return RoughPlastic(albedo, mat.alpha, mat.extIOR, mat.intIOR).sample(bs, u);
            case BSDF_ID_DIFFUSE:
            default:                      return Diffuse(albedo).sample(bs, u);
        }
    }

    // Evaluate BSDF f(wo, wi)
    HD static Color3f eval(const Material& mat, const SurfaceInteraction& si,
                           const Vector3f& wo_local) 
    {
        Color3f f; float p;
        eval_pdf(mat, si, wo_local, f, p);
        return f;
    }

    // Evaluate PDF p(wo | wi)
    HD static float pdf(const Material& mat, const SurfaceInteraction& si,
                        const Vector3f& wo_local) 
    {
        Color3f f; float p;
        eval_pdf(mat, si, wo_local, f, p);
        return p;
    }

    // Simultaneously evaluate BSDF f(wo, wi) and its PDF p(wo | wi)
    HD static void eval_pdf(const Material& mat, const SurfaceInteraction& si,
                            const Vector3f& wo_local,
                            Color3f& f_out, float& pdf_out) 
    {
        if (is_delta(mat.type)) {
            f_out   = Color3f(0.f);
            pdf_out = 0.f;
            return;
        }

        BSDFSample bs;
        bs.wi         = si.to_local(si.wi);
        bs.front_face = si.front_face;
        bs.wo         = wo_local;
        const Color3f albedo = get_albedo(mat, si);

        switch (mat.type) {
            case BSDF_ID_MICROFACET:      return Microfacet(albedo, mat.alpha, mat.extIOR, mat.intIOR, mat.isConductor, mat.conductorEta, mat.conductorK, mat.specular).eval_pdf(bs, f_out, pdf_out);
            case BSDF_ID_ROUGHDIELECTRIC: return RoughDielectric(albedo, mat.alpha, mat.extIOR, mat.intIOR).eval_pdf(bs, f_out, pdf_out);
            case BSDF_ID_ROUGHCONDUCTOR:  return RoughConductor(albedo, mat.alpha, mat.extIOR, mat.conductorEta, mat.conductorK, mat.specular).eval_pdf(bs, f_out, pdf_out);
            case BSDF_ID_ROUGHPLASTIC:    return RoughPlastic(albedo, mat.alpha, mat.extIOR, mat.intIOR).eval_pdf(bs, f_out, pdf_out);
            case BSDF_ID_DIFFUSE:
            default:                      return Diffuse(albedo).eval_pdf(bs, f_out, pdf_out);
        }
    }
};

FUTABA_NAMESPACE_END
