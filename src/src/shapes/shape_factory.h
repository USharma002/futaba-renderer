#pragma once

#include <string>
#include <vector>
#include "types.cuh"
#include "triangle.cuh"

namespace futaba {
struct LoadedScene;

bool appendMeshGeometry(const std::string& meshName,
                       int materialId,
                       int emitterId,
                       const Matrix4f& transform,
                       const Matrix4f& normalTransform,
                       const std::vector<Triangle>& localTriangles,
                       LoadedScene& out);

bool appendRectangleShape(const std::string& meshName,
                         int materialId,
                         int emitterId,
                         const Matrix4f& transform,
                         const Matrix4f& normalTransform,
                         LoadedScene& out);

bool appendSphereShape(const std::string& meshName,
                      float radius,
                      const Point3f& center,
                      int materialId,
                      int emitterId,
                      const Matrix4f& transform,
                      const Matrix4f& normalTransform,
                      LoadedScene& out);

bool appendDiskShape(const std::string& meshName,
                    int materialId,
                    int emitterId,
                    const Matrix4f& transform,
                    const Matrix4f& normalTransform,
                    LoadedScene& out);

bool appendCubeShape(const std::string& meshName,
                    int materialId,
                    int emitterId,
                    const Matrix4f& transform,
                    const Matrix4f& normalTransform,
                    LoadedScene& out);

} // namespace futaba
