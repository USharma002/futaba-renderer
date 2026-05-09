#include "ray.cuh"
#include "common.cuh"
#include "../bsdfs/diffuse.cuh"

namespace futaba {

struct Sphere {
    Point3f center;
    float radius;
    int material_id;
    Diffuse bsdf;

    HD bool intersect(const Ray& r, float t_min, float t_max, SurfaceIntersection& rec) const {
        Vector3f oc = r.o - center;
        float a = dot(r.d, r.d);
        float b = 2.0f * dot(oc, r.d);
        float c = dot(oc, oc) - radius * radius;
        float discriminant = b * b - 4.0f * a * c;

        if (discriminant < 0.0f) return false;

        float sqrtd = sqrtf(discriminant);
        float root = (-b - sqrtd) / (2.0f * a);
        
        if (root < t_min || root > t_max) {
            root = (-b + sqrtd) / (2.0f * a);
            if (root < t_min || root > t_max) return false;
        }

        // We have a hit! Fill out the record.
        rec.t = root;
        rec.p = r(rec.t);
        rec.n = normalize((rec.p - center) / radius);
        rec.wi = -r.d;
        rec.shape_id = -1;
        rec.material_id = material_id;
        rec.albedo = bsdf.albedo;
        rec.front_face = dot(r.d, rec.n) < 0.0f;
        Vector3f frame_n = rec.front_face ? rec.n : -rec.n;
        rec.set_frame_from_normal(frame_n);
        
        return true;
    }
};

}