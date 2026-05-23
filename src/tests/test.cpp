#include <iomanip>
#include <iostream>

#include "bvh.cuh"
#include "bsdf_sample.cuh"
#include "ray.cuh"
#include "emitter_sampler.cuh"
#include "envmap.cuh"
#include "frame.cuh"
#include "launch_params.h"
#include "path.cuh"
#include "phong.cuh"
#include "surface_interaction.cuh"
#include "volpath.cuh"
#include "material.cuh"
#include "distribution.cuh"
#include "scene.cuh"
#include "perspective.cuh"
#include "sphere.cuh"
#include "triangle.cuh"

namespace {

template <typename T>
void print_type(const char* name) {
    std::cout << std::left << std::setw(34) << name
              << " size=" << std::setw(4) << sizeof(T)
              << " align=" << alignof(T) << '\n';
}

} // namespace

int main() {
    using namespace futaba;

    print_type<Vector2f>("Vector2f");
    print_type<Point2f>("Point2f");
    print_type<Vector3f>("Vector3f");
    print_type<Point3f>("Point3f");
    print_type<Color3f>("Color3f");
    print_type<Normal3f>("Normal3f");

    print_type<Ray3f>("Ray3f");
    print_type<Frame>("Frame");
    print_type<BSDFSample>("BSDFSample");
    print_type<Material>("Material");
    print_type<SurfaceIntersection>("SurfaceIntersection");
    print_type<Triangle>("Triangle");
    print_type<Sphere>("Sphere");
    print_type<AABB>("AABB");
    print_type<BVHNode>("BVHNode");
    print_type<BVH>("BVH");

    print_type<EnvironmentMapEmitter>("EnvironmentMapEmitter");
    print_type<EmitterSample>("EmitterSample");
    print_type<EmitterGPU>("EmitterGPU");
    print_type<Scene>("Scene");
    print_type<PerspectiveCamera>("PerspectiveCamera");

    print_type<Distribution1D>("Distribution1D");
    print_type<Distribution2D>("Distribution2D");
    print_type<Path<UniformEmitterSampler>>("Path<UniformEmitterSampler>");
    print_type<VolumetricPath>("VolumetricPath");
    print_type<Phong>("Phong");
    print_type<LaunchParams>("LaunchParams");

    return 0;
}