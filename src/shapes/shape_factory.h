#pragma once

#include <string>
#include <vector>
#include "types.cuh"
#include "triangle.cuh"

FUTABA_NAMESPACE_BEGIN

struct CPUScene;

bool appendMeshGeometry(const std::string& meshName,
                       int materialId,
                       int emitterId,
                       const Matrix4f& transform,
                       const Matrix4f& normalTransform,
                       const std::vector<Triangle>& localTriangles,
                       CPUScene& out);

bool appendRectangleShape(const std::string& meshName,
                         int materialId,
                         int emitterId,
                         const Matrix4f& transform,
                         const Matrix4f& normalTransform,
                         CPUScene& out);

bool appendSphereShape(const std::string& meshName,
                      float radius,
                      const Point3f& center,
                      int materialId,
                      int emitterId,
                      const Matrix4f& transform,
                      const Matrix4f& normalTransform,
                      CPUScene& out);

bool appendDiskShape(const std::string& meshName,
                    int materialId,
                    int emitterId,
                    const Matrix4f& transform,
                    const Matrix4f& normalTransform,
                    CPUScene& out);

bool appendCubeShape(const std::string& meshName,
                    int materialId,
                    int emitterId,
                    const Matrix4f& transform,
                    const Matrix4f& normalTransform,
                    CPUScene& out);

FUTABA_NAMESPACE_END