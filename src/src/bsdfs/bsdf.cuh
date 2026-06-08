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

namespace futaba {

struct BSDF {
    // Check if the BSDF is Dirac-delta
    HD static bool is_delta(BSDFType type) {
        return type == BSDF_ID_MIRROR || type == BSDF_ID_DIELECTRIC || type == BSDF_ID_THINDIELECTRIC;
    }

    // Evaluate the albedo (including texture lookup if applicable)
    HD static Color3f get_albedo(const Material& mat, const SurfaceIntersection& si) {
        Color3f albedo = mat.albedo;
#ifdef __CUDA_ARCH__
        if (mat.texObj != 0) {
            float4 texVal = tex2D<float4>(mat.texObj, si.uv.x, si.uv.y);
            albedo = mat.albedo * Color3f(texVal.x, texVal.y, texVal.z);
        }
#endif
        return albedo;
    }

private:
    // Construct the concrete BSDF object for mat.type and call bsdf.sample(bs, s2).
    // ThinDielectric and Null are handled before this is called.
    HD static Color3f dispatch_sample(const Material& mat, const Color3f& albedo,
                                      BSDFSample& bs, const Point2f& s2) {
        switch (mat.type) {
            case BSDF_ID_MICROFACET: {
                Microfacet bsdf(albedo, mat.alpha, mat.extIOR, mat.intIOR,
                                mat.isConductor, mat.conductorEta, mat.conductorK, mat.specular);
                return bsdf.sample(bs, s2);
            }
            case BSDF_ID_DIELECTRIC: {
                Dielectric bsdf(albedo, mat.intIOR);
                return bsdf.sample(bs, s2);
            }
            case BSDF_ID_MIRROR: {
                Mirror bsdf(albedo);
                return bsdf.sample(bs, s2);
            }
            case BSDF_ID_ROUGHDIELECTRIC: {
                RoughDielectric bsdf(albedo, mat.alpha, mat.extIOR, mat.intIOR);
                return bsdf.sample(bs, s2);
            }
            case BSDF_ID_ROUGHCONDUCTOR: {
                RoughConductor bsdf(albedo, mat.alpha, mat.extIOR, mat.conductorEta, mat.conductorK, mat.specular);
                return bsdf.sample(bs, s2);
            }
            case BSDF_ID_ROUGHPLASTIC: {
                RoughPlastic bsdf(albedo, mat.alpha, mat.extIOR, mat.intIOR);
                return bsdf.sample(bs, s2);
            }
            default: { // BSDF_ID_DIFFUSE
                Diffuse bsdf(albedo);
                return bsdf.sample(bs, s2);
            }
        }
    }

    // Construct the concrete BSDF object for mat.type and call bsdf.eval/pdf.
    // ThinDielectric and Null are handled before this is called.
    HD static void dispatch_eval_pdf(const Material& mat, const Color3f& albedo,
                                     const BSDFSample& bs,
                                     Color3f& f_out, float& pdf_out) {
        switch (mat.type) {
            case BSDF_ID_MICROFACET: {
                Microfacet bsdf(albedo, mat.alpha, mat.extIOR, mat.intIOR,
                                mat.isConductor, mat.conductorEta, mat.conductorK, mat.specular);
                f_out   = bsdf.eval(bs);
                pdf_out = bsdf.pdf(bs);
                return;
            }
            case BSDF_ID_DIELECTRIC: {
                Dielectric bsdf(albedo, mat.intIOR);
                f_out   = bsdf.eval(bs);
                pdf_out = bsdf.pdf(bs);
                return;
            }
            case BSDF_ID_MIRROR: {
                Mirror bsdf(albedo);
                f_out   = bsdf.eval(bs);
                pdf_out = bsdf.pdf(bs);
                return;
            }
            case BSDF_ID_ROUGHDIELECTRIC: {
                RoughDielectric bsdf(albedo, mat.alpha, mat.extIOR, mat.intIOR);
                f_out   = bsdf.eval(bs);
                pdf_out = bsdf.pdf(bs);
                return;
            }
            case BSDF_ID_ROUGHCONDUCTOR: {
                RoughConductor bsdf(albedo, mat.alpha, mat.extIOR, mat.conductorEta, mat.conductorK, mat.specular);
                f_out   = bsdf.eval(bs);
                pdf_out = bsdf.pdf(bs);
                return;
            }
            case BSDF_ID_ROUGHPLASTIC: {
                RoughPlastic bsdf(albedo, mat.alpha, mat.extIOR, mat.intIOR);
                f_out   = bsdf.eval(bs);
                pdf_out = bsdf.pdf(bs);
                return;
            }
            default: { // BSDF_ID_DIFFUSE
                Diffuse bsdf(albedo);
                f_out   = bsdf.eval(bs);
                pdf_out = bsdf.pdf(bs);
                return;
            }
        }
    }

public:
    // Sample the BSDF
    HD static Color3f sample(const Material& mat, const SurfaceIntersection& si,
                             BSDFSample& bs, const Point2f& s2) {
        bs.wi         = si.to_local(si.wi);
        bs.front_face = si.front_face;

        const Color3f albedo = get_albedo(mat, si);

        // Passthrough types that don't share the standard eval/pdf interface
        if (mat.type == BSDF_ID_THINDIELECTRIC) {
            ThinDielectric bsdf(albedo, mat.intIOR);
            return bsdf.sample(bs, s2);
        }
        if (mat.type == BSDF_ID_NULL) {
            bs.wo           = -bs.wi;
            bs.sampled_type = BSDF_ID_NULL;
            bs.pdf          = 1.0f;
            bs.eta          = 1.0f;
            return Color3f(1.0f);
        }

        return dispatch_sample(mat, albedo, bs, s2);
    }

    // Evaluate BSDF and its PDF
    HD static void eval_pdf(const Material& mat, const SurfaceIntersection& si,
                            const Vector3f& wo_local,
                            Color3f& f_out, float& pdf_out) {
        // Delta/passthrough types have no continuous eval or pdf
        if (mat.type == BSDF_ID_THINDIELECTRIC || mat.type == BSDF_ID_NULL) {
            f_out   = Color3f(0.f);
            pdf_out = 0.f;
            return;
        }

        BSDFSample bs;
        bs.wi         = si.to_local(si.wi);
        bs.front_face = si.front_face;
        bs.wo         = wo_local;

        const Color3f albedo = get_albedo(mat, si);

        dispatch_eval_pdf(mat, albedo, bs, f_out, pdf_out);
    }
};

} // namespace futaba
