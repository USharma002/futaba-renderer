#pragma once

#include "types.cuh"
#include "ray.cuh"

FUTABA_NAMESPACE_BEGIN

struct AABB {
    Point3f min;
    Point3f max;

    HD AABB() : min(Point3f(1e30f)), max(Point3f(-1e30f)) {}
    HD AABB(const Point3f& p) : min(p), max(p) {}
    HD AABB(const Point3f& min, const Point3f& max) : min(min), max(max) {}

    // Expand the box to enclose a point
    HD void grow(const Point3f& p) {
        min.x = FAST_MIN(min.x, p.x);
        min.y = FAST_MIN(min.y, p.y);
        min.z = FAST_MIN(min.z, p.z);
        max.x = FAST_MAX(max.x, p.x);
        max.y = FAST_MAX(max.y, p.y);
        max.z = FAST_MAX(max.z, p.z);
    }

    // Expand the box to enclose another box
    HD void grow(const AABB& other) {
        min.x = FAST_MIN(min.x, other.min.x);
        min.y = FAST_MIN(min.y, other.min.y);
        min.z = FAST_MIN(min.z, other.min.z);
        max.x = FAST_MAX(max.x, other.max.x);
        max.y = FAST_MAX(max.y, other.max.y);
        max.z = FAST_MAX(max.z, other.max.z);
    }

    HD void expand(const Point3f& p) { grow(p); }
    HD void expand(const AABB& other) { grow(other); }

    HD Vector3f extent() const {
        return max - min;
    }

    // Centroid of the bounding box (useful for BVH split heuristics)
    HD Point3f centroid() const {
        return (min + max) * 0.5f;
    }

    // Ray-AABB intersection test with precomputed inverse direction. Safely handles NaNs.
    HD bool intersectDist(const Ray& ray, const Vector3f& dRcp, float tMin, float tMax, float& dist) const {
        float tx1 = (min.x - ray.o.x) * dRcp.x;
        float tx2 = (max.x - ray.o.x) * dRcp.x;
        float tNear = tx1 < tx2 ? tx1 : tx2;
        float tFar = tx1 > tx2 ? tx1 : tx2;
        tMin = tNear > tMin ? tNear : tMin;
        tMax = tFar < tMax ? tFar : tMax;

        float ty1 = (min.y - ray.o.y) * dRcp.y;
        float ty2 = (max.y - ray.o.y) * dRcp.y;
        tNear = ty1 < ty2 ? ty1 : ty2;
        tFar = ty1 > ty2 ? ty1 : ty2;
        tMin = tNear > tMin ? tNear : tMin;
        tMax = tFar < tMax ? tFar : tMax;

        float tz1 = (min.z - ray.o.z) * dRcp.z;
        float tz2 = (max.z - ray.o.z) * dRcp.z;
        tNear = tz1 < tz2 ? tz1 : tz2;
        tFar = tz1 > tz2 ? tz1 : tz2;
        tMin = tNear > tMin ? tNear : tMin;
        tMax = tFar < tMax ? tFar : tMax;

        if (tMax >= tMin) {
            dist = tMin;
            return true;
        }
        return false;
    }

    // Ray-AABB intersection test computing inverse direction on-demand.
    HD bool intersectDist(const Ray& ray, float tMin, float tMax, float& dist) const {
        Vector3f dRcp(1.0f / ray.d.x, 1.0f / ray.d.y, 1.0f / ray.d.z);
        return intersectDist(ray, dRcp, tMin, tMax, dist);
    }

    HD bool intersect(const Ray& ray, float tMin, float tMax) const {
        float dist;
        return intersectDist(ray, tMin, tMax, dist);
    }
};

FUTABA_NAMESPACE_END
