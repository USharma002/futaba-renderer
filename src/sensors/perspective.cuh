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
    float fovY;
    float tanHalfFovY;
    float aspectRatio;

    // Thin-lens depth-of-field parameters.
    float focusDistance;
    float apertureRadius;

    // Default constructor
    HD PerspectiveCamera() {
        init(Point3f(0, 0, 5), Point3f(0, 0, 0), Vector3f(0, 1, 0), 45.0f,
             16.0f / 9.0f);
    }

    HD PerspectiveCamera(float fovY, float aspect) {
        init(Point3f(0, 0, 5), Point3f(0, 0, 0), Vector3f(0, 1, 0), fovY,
             aspect);
    }

    HD PerspectiveCamera(Point3f pos, Point3f target, Vector3f up, float fovY,
                         float aspect) {
        init(pos, target, up, fovY, aspect);
    }

    // Fully initialises the camera. Must be called whenever position, target, up, FOV,
    // or aspect ratio change. All other helpers (setPosition, setTarget, setUp) delegate
    // here to ensure the orthonormal frame is always consistent.
    HD void init(Point3f pos, Point3f target, Vector3f up, float fovY,
                 float aspect, float focusDist = 5.0f,
                 float aperture = 0.0f) {
        position    = pos;
        aspectRatio = aspect;
        this->fovY  = fovY;
        focusDistance = focusDist;
        apertureRadius = aperture;

        forward = normalize(target - position);
        right   = normalize(cross(forward, up));
        trueUp  = cross(right, forward); // already unit length (forward and right are orthonormal)

        tanHalfFovY = tanf(degToRad(fovY) / 2.0f);
    }

    // Generates a world-space ray through normalised image-plane coordinate (u, v).
    // u, v ∈ [0, 1]. Runs on the GPU millions of times per frame.
    HD Ray3f sampleRay(float u, float v, Sampler &sampler) const {
        
        float px = (2.0f * u - 1.0f) * aspectRatio * tanHalfFovY;
        float py = (1.0f - 2.0f * v) * tanHalfFovY;

        // No matrix inversion - use precomputed orthonormal axes.
        Vector3f pinholeDir = normalize(right * px + trueUp * py + forward);

        // If aperture is 0, this reduces to a simple pinhole ray.
        if (apertureRadius <= 0.0f) {
            return Ray3f(position, pinholeDir);
        }

        // Sample a point on the lens (disk) and compute the ray direction towards the focus point.
        Point2f disk = sampler.next2D();
        float r = sqrtf(disk.x);
        float theta = 2.0f * static_cast<float>(M_PI) * disk.y;
        float lensX = r * cosf(theta) * apertureRadius;
        float lensY = r * sinf(theta) * apertureRadius;

        float focusT = focusDistance / dot(pinholeDir, forward);
        Point3f focusPoint = position + pinholeDir * focusT;
        Point3f lensOrigin = position + right * lensX + trueUp * lensY;
        Vector3f dir = normalize(focusPoint - lensOrigin);
        return Ray3f(lensOrigin, dir);
    }

    // -----------------------------------------------------------------------
    // Helpers - each one rebuilds the full frame so the invariant is preserved.
    // -----------------------------------------------------------------------

    // Change FOV; does not affect the viewing direction or axes.
    HD void setFov(float newFovY) {
        fovY = newFovY;
        tanHalfFovY = tanf(degToRad(newFovY) / 2.0f);
    }

    HD void setFocusDistance(float newFocusDistance) {
        focusDistance = newFocusDistance;
    }

    HD void setApertureRadius(float newApertureRadius) {
        apertureRadius = newApertureRadius;
    }

    // Reposition the camera origin; recomputes axes from the new position.
    // The current forward and trueUp are used as hints for target and up respectively.
    HD void setPosition(const Point3f& pos) {
        Point3f target = position + forward; // keep the same look direction
        init(pos, target, trueUp, fovY, aspectRatio, focusDistance,
             apertureRadius);
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
