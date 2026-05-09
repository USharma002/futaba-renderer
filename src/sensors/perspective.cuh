#pragma once

#include "types.cuh"
#include "common.cuh"
#include "sampler.cuh"
#include "ray.cuh"
#include <cmath>

namespace futaba {

class PerspectiveCamera {
public: 
    Point3f position;
    
    // Precomputed Axes (World Space)
    Vector3f forward;
    Vector3f right;
    Vector3f trueUp;
    
    // Precomputed FOV values
    float tanHalfFovY;
    float aspectRatio;

    // Default constructor
    HD PerspectiveCamera() {
        init(Point3f(0, 0, 5), Point3f(0, 0, 0), Vector3f(0, 1, 0), 45.0f, 16.0f / 9.0f);
    }

    HD PerspectiveCamera(float fovY, float aspect) {
        init(Point3f(0, 0, 5), Point3f(0, 0, 0), Vector3f(0, 1, 0), fovY, aspect);
    }
    
    // Parameterized constructor
    HD PerspectiveCamera(Point3f pos, Point3f target, Vector3f up, float fovY, float aspect) {
        init(pos, target, up, fovY, aspect);
    }

    // Call this whenever the camera moves or the window resizes
    HD void init(Point3f pos, Point3f target, Vector3f up, float fovY, float aspect) {
        position = pos;
        aspectRatio = aspect;
        
        // Calculate camera axes using our global normalize() and cross() functions
        forward = normalize(target - position);
        right = normalize(cross(forward, up));
        trueUp = cross(right, forward);

        // Precompute the tangent so we don't do trigonometry per-pixel
        tanHalfFovY = tanf(degToRad(fovY) / 2.0f);
    }
    
    // This runs on the GPU millions of times per frame!
    HD Ray3f sampleRay(float u, float v) const {
        // (2u - 1) remaps 0..1 to -1..1
        // (1 - 2v) remaps 0..1 to 1..-1 (flips Y axis so +Y is up)
        float px = (2.0f * u - 1.0f) * aspectRatio * tanHalfFovY;
        float py = (1.0f - 2.0f * v) * tanHalfFovY;

        // Instantly build the world-space ray direction using our precomputed axes!
        // No matrix inversion required.
        Vector3f rayDirWorld = normalize(right * px + trueUp * py + forward);

        return Ray3f(position, rayDirWorld);
    }

    // Helper to update FOV
    HD void setFov(float newFovY) {
        tanHalfFovY = tanf(degToRad(newFovY) / 2.0f);
    }
    
    // Helper to set camera position
    HD void setPosition(const Point3f& pos) {
        position = pos;
    }
    
    // Helper to set camera target (and recompute axes)
    HD void setTarget(const Point3f& target) {
        forward = normalize(target - position);
    }
    
    // Helper to set camera up vector (and recompute axes)
    HD void setUp(const Vector3f& up) {
        right = normalize(cross(forward, up));
        trueUp = cross(right, forward);
    }
};

} // namespace futaba