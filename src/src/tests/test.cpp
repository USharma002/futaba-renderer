#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

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

#include "ppg/stree.h"
#include "ppg/dtree.cuh"

// Bring types under scope to allow anonymous namespace definitions to pass compiler checks
using namespace futaba;


template <typename T>
void print_type(const char* name) {
    std::cout << std::left << std::setw(34) << name
              << " size=" << std::setw(4) << sizeof(T)
              << " align=" << alignof(T) << '\n';
}


void printStructSize(){
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
    print_type<Path>("Path");
    print_type<VolumetricPath>("VolumetricPath");
    print_type<Phong>("Phong");
    print_type<LaunchParams>("LaunchParams");
    print_type<QuadTreeNode>("QuadTreeNode");
    print_type<DTree>("DTree");

    std::cout << "\nMaterial offsets:\n";
    std::cout << "  albedo:       " << offsetof(Material, albedo) << "\n";
    std::cout << "  specular:     " << offsetof(Material, specular) << "\n";
    std::cout << "  emission:     " << offsetof(Material, emission) << "\n";
    std::cout << "  conductorEta: " << offsetof(Material, conductorEta) << "\n";
    std::cout << "  conductorK:   " << offsetof(Material, conductorK) << "\n";
    std::cout << "  alpha:        " << offsetof(Material, alpha) << "\n";
    std::cout << "  extIOR:       " << offsetof(Material, extIOR) << "\n";
    std::cout << "  intIOR:       " << offsetof(Material, intIOR) << "\n";
    std::cout << "  isConductor:  " << offsetof(Material, isConductor) << "\n";
    std::cout << "  type:         " << offsetof(Material, type) << "\n";
    std::cout << "  texObj:       " << offsetof(Material, texObj) << "\n";
}

int main(int argc, char** argv) {
    bool showTree = true;
    bool showSizes = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--sizes") {
            showTree = false;
            showSizes = true;
        } else if (arg == "--both") {
            showTree = true;
            showSizes = true;
        } else if (arg == "--tree") {
            showTree = true;
        }
    }

    if (showSizes) {
        std::cout << "\nType sizes:\n";
        printStructSize();
    }

    return 0;
}