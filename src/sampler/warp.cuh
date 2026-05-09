#pragma once

#include "types.cuh"
#include "common.cuh"
#include <cmath>

// Warp functions for importance sampling
// All points have [0, 1]^d uniform samples
struct Warp {
    // Uniformly sample a vector on the unit hemisphere around the pole (0,0,1) with respect to projected solid angles
    HD static Vector3f square_to_cosine_hemisphere(const Point2f& sample2) {
        float phi = 2.0f * M_PI * sample2.x;
        float cos_theta = sqrtf(sample2.y);
        float sin_theta = sqrtf(1.0f - sample2.y);
        
        return Vector3f(
            sin_theta * cosf(phi),
            sin_theta * sinf(phi),
            cos_theta
        );
    }
    
    // Probability density of squareToCosineHemisphere()
    HD static float square_to_cosine_hemisphere_pdf(const Vector3f& wo) {
        float cos_theta = wo.z;
        if (cos_theta <= 0.0f) return 0.0f;
        return cos_theta * INV_PI;
    }
    
    // Uniform hemisphere sampling
    HD static Vector3f square_to_uniform_hemisphere(const Point2f& sample2) {
        float phi = 2.0f * M_PI * sample2.x;
        float cos_theta = 1.0f - sample2.y;
        float sin_theta = sqrtf(sample2.y * (2.0f - sample2.y));
        
        return Vector3f(
            sin_theta * cosf(phi),
            sin_theta * sinf(phi),
            cos_theta
        );
    }
    
    // Probability density for uniform hemisphere sampling
    HD static float square_to_uniform_hemisphere_pdf(const Vector3f&) {
        return INV_PI / 2.0f;
    }
    
};
