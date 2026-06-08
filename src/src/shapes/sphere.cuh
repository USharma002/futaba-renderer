#include "ray.cuh"
#include "common.cuh"
#include "diffuse.cuh"

namespace futaba {

struct Sphere {
    float radius; // Radius of the sphere (4 bytes)
    int material_id = -1; // Material ID (4 bytes)
    Point3f center; // Center of the sphere (12 bytes)
    
    HD bool intersect(const Ray& r, float t_min, float t_max, SurfaceIntersection& rec) const {
        Vector3f oc = r.o - center;

        // Solve the simplified quadratic equation for normalized ray direction:
        // t^2 + 2*half_b*t + c = 0
        float half_b = dot(oc, r.d);
        float c = dot(oc, oc) - radius * radius;
        float discriminant = half_b * half_b - c;

        if (discriminant < 0.0f) return false;

        float sqrtd = sqrtf(discriminant);
        float root = -half_b - sqrtd;
        
        if (root < t_min || root > t_max) {
            root = -half_b + sqrtd;
            if (root < t_min || root > t_max) return false;
        }

        // We have a hit! Fill out the record.
        rec.t = root;
        rec.p = r(rec.t);
        rec.n = (rec.p - center) / radius; // already unit length (dist is radius)
        rec.wi = -r.d;
        rec.shape_id = -1;
        rec.material_id = material_id;
        rec.front_face = dot(r.d, rec.n) < 0.0f;
        
        Vector3f frame_n = rec.front_face ? rec.n : -rec.n;
        rec.set_frame_from_normal(frame_n);
        
        return true;
    }
};

}