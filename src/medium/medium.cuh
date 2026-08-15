#pragma once

#include "types.cuh"
#include "common.cuh"
#include "phase.cuh"
#include <cmath>

FUTABA_NAMESPACE_BEGIN

enum MediumType {
    MEDIUM_VACUUM      = 0,
    MEDIUM_HOMOGENEOUS = 1
};

struct MediumInteraction {
    Point3f  p        = Point3f(0.f);
    Vector3f wo       = Vector3f(0.f); // Direction pointing back to ray origin (-ray.d)
    float    t        = 0.f;
    bool     isValid  = false;
};

struct Medium {
    Color3f    sigma_a = Color3f(0.f);
    Color3f    sigma_s = Color3f(0.f);
    Color3f    sigma_t = Color3f(0.f);
    float      g       = 0.0f;
    MediumType type    = MEDIUM_VACUUM;

    HD Medium() : sigma_a(0.f), sigma_s(0.f), sigma_t(0.f), g(0.f), type(MEDIUM_VACUUM) {}

    HD Medium(const Color3f& sa, const Color3f& ss, float g_val = 0.f)
        : sigma_a(sa), sigma_s(ss), sigma_t(sa + ss), g(g_val), type(MEDIUM_HOMOGENEOUS) {}

    HD bool is_active() const {
        return type != MEDIUM_VACUUM && (sigma_t.x > 1e-6f || sigma_t.y > 1e-6f || sigma_t.z > 1e-6f);
    }

    // Evaluate spectral beam transmittance over distance dist
    HD Color3f eval_transmittance(float dist) const {
        if (!is_active() || dist <= 0.f) return Color3f(1.f);
        return Color3f(
            expf(-sigma_t.x * dist),
            expf(-sigma_t.y * dist),
            expf(-sigma_t.z * dist)
        );
    }

    // Sample an interaction distance along a ray segment [0, max_t]
    // Returns true if scattering occurred within the medium before max_t
    HD bool sample_distance(const Ray& ray, float max_t, float u, int channel,
                            MediumInteraction& mi, Color3f& tr_weight, float& pdf_out) const
    {
        if (!is_active() || max_t <= 0.f) {
            tr_weight = Color3f(1.f);
            pdf_out   = 1.0f;
            mi.isValid = false;
            return false;
        }

        // Spectral channel selection for delta/exponential tracking
        float st = (channel == 0) ? sigma_t.x : ((channel == 1) ? sigma_t.y : sigma_t.z);
        if (st <= 1e-6f) st = 1e-6f;

        float sampled_t = -log1pf(-fminf(u, 0.999999f)) / st;

        if (sampled_t < max_t) {
            // Medium in-scattering event
            mi.t       = sampled_t;
            mi.p       = ray.o + ray.d * sampled_t;
            mi.wo      = -ray.d;
            mi.isValid = true;

            Color3f tr      = eval_transmittance(sampled_t);
            Color3f density = sigma_t * tr;
            pdf_out         = (density.x + density.y + density.z) / 3.0f;
            tr_weight       = (pdf_out > 0.f) ? (tr * sigma_s / pdf_out) : Color3f(0.f);
            return true;
        } else {
            // Reached surface without medium scattering
            mi.isValid = false;
            Color3f tr = eval_transmittance(max_t);
            pdf_out    = (tr.x + tr.y + tr.z) / 3.0f;
            tr_weight  = (pdf_out > 0.f) ? (tr / pdf_out) : Color3f(1.f);
            return false;
        }
    }
};

FUTABA_NAMESPACE_END
