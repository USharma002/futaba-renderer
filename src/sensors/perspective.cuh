#pragma once

#include "types.cuh"
#include "common.cuh"
#include "sampler.cuh"
#include "ray.cuh"
#include <cmath>

namespace futaba {

class PerspectiveCamera {
public:
    Point3f  position;

    // Precomputed orthonormal axes (world space).
    // Invariant: right, trueUp, forward are always mutually orthogonal unit vectors.
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

    HD PerspectiveCamera(Point3f pos, Point3f target, Vector3f up, float fovY, float aspect) {
        init(pos, target, up, fovY, aspect);
    }

    // Fully initialises the camera. Must be called whenever position, target, up, FOV,
    // or aspect ratio change. All other helpers (setPosition, setTarget, setUp) delegate
    // here to ensure the orthonormal frame is always consistent.
    HD void init(Point3f pos, Point3f target, Vector3f up, float fovY, float aspect) {
        position    = pos;
        aspectRatio = aspect;

        forward = normalize(target - position);
        right   = normalize(cross(forward, up));
        trueUp  = cross(right, forward); // already unit length (forward and right are orthonormal)

        tanHalfFovY = tanf(degToRad(fovY) / 2.0f);
    }

    // Generates a world-space ray through normalised image-plane coordinate (u, v).
    // u, v ∈ [0, 1]. Runs on the GPU millions of times per frame.
    HD Ray3f sampleRay(float u, float v) const {
        
        float px = (2.0f * u - 1.0f) * aspectRatio * tanHalfFovY;
        float py = (1.0f - 2.0f * v) * tanHalfFovY;

        // No matrix inversion - use precomputed orthonormal axes.
        Vector3f dir = normalize(right * px + trueUp * py + forward);
        return Ray3f(position, dir);
    }

    // -----------------------------------------------------------------------
    // Helpers - each one rebuilds the full frame so the invariant is preserved.
    // -----------------------------------------------------------------------

    // Change FOV; does not affect the viewing direction or axes.
    HD void setFov(float newFovY) {
        tanHalfFovY = tanf(degToRad(newFovY) / 2.0f);
    }

    // Reposition the camera origin; recomputes axes from the new position.
    // The current forward and trueUp are used as hints for target and up respectively.
    HD void setPosition(const Point3f& pos) {
        Point3f target = position + forward; // keep the same look direction
        init(pos, target, trueUp, 0.f, aspectRatio);
    }

    // Repoint the camera at a new target, recomputing the full orthonormal frame.
    // Uses the current trueUp as the "up hint". If forward becomes parallel to trueUp
    // (gimbal lock) the result is undefined - use full init() in that case.
    HD void setTarget(const Point3f& target) {
        forward = normalize(target - position);
        right   = normalize(cross(forward, trueUp));
        trueUp  = cross(right, forward);
    }

    // Replace the up hint and recompute right / trueUp from the current forward.
    HD void setUp(const Vector3f& up) {
        right  = normalize(cross(forward, up));
        trueUp = cross(right, forward);
    }
};

} // namespace futaba
