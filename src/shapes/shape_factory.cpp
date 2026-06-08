#include "shape_factory.h"
#include "scene_loader.h"
#include "common.cuh"
#include <algorithm>
#include <cmath>

namespace futaba {

bool appendMeshGeometry(const std::string& meshName,
                       int materialId,
                       int emitterId,
                       const Matrix4f& transform,
                       const Matrix4f& normalTransform,
                       const std::vector<Triangle>& localTriangles,
                       LoadedScene& out)
{
    const uint32_t meshTriangleStart = (uint32_t)out.triangles.size();
    const int meshId = (int)out.meshes.size();

    Point3f boundsMin(1e30f, 1e30f, 1e30f);
    Point3f boundsMax(-1e30f, -1e30f, -1e30f);

    for (const auto& localT : localTriangles) {
        Triangle t = localT;
        t.p0 = transform * localT.p0;
        t.p1 = transform * localT.p1;
        t.p2 = transform * localT.p2;

        if (t.has_normals) {
            t.n0 = normalize(normalTransform * localT.n0);
            t.n1 = normalize(normalTransform * localT.n1);
            t.n2 = normalize(normalTransform * localT.n2);
        }

        t.material_id = materialId;
        t.mesh_id = meshId;

        for (const auto& p : {t.p0, t.p1, t.p2}) {
            boundsMin.x = std::min(boundsMin.x, p.x);
            boundsMin.y = std::min(boundsMin.y, p.y);
            boundsMin.z = std::min(boundsMin.z, p.z);
            boundsMax.x = std::max(boundsMax.x, p.x);
            boundsMax.y = std::max(boundsMax.y, p.y);
            boundsMax.z = std::max(boundsMax.z, p.z);
        }

        out.triangles.push_back(t);
    }

    MeshInstance meshInst;
    meshInst.name          = meshName;
    meshInst.materialId    = materialId;
    meshInst.triangleStart = meshTriangleStart;
    meshInst.triangleCount = (uint32_t)localTriangles.size();
    meshInst.transform     = transform;
    meshInst.emitterType   = (emitterId >= 0) ? EmitterType::Area : EmitterType::None;
    meshInst.emitterId     = emitterId;
    meshInst.boundingBoxMin = boundsMin;
    meshInst.boundingBoxMax = boundsMax;

    out.meshes.push_back(meshInst);
    return true;
}

bool appendRectangleShape(const std::string& meshName,
                         int materialId,
                         int emitterId,
                         const Matrix4f& transform,
                         const Matrix4f& normalTransform,
                         LoadedScene& out)
{
    std::vector<Triangle> localTriangles;
    localTriangles.reserve(2);

    const Point3f local[4] = {
        Point3f(-1.f, -1.f, 0.f),
        Point3f( 1.f, -1.f, 0.f),
        Point3f( 1.f,  1.f, 0.f),
        Point3f(-1.f,  1.f, 0.f)
    };

    Vector3f n(0.f, 0.f, 1.f);

    Triangle t0;
    t0.p0 = local[0]; t0.p1 = local[1]; t0.p2 = local[2];
    t0.n0 = n; t0.n1 = n; t0.n2 = n;
    t0.has_normals = true;
    t0.uv0 = Point2f(0.f, 0.f); t0.uv1 = Point2f(1.f, 0.f); t0.uv2 = Point2f(1.f, 1.f);
    t0.has_uvs = true;
    localTriangles.push_back(t0);

    Triangle t1;
    t1.p0 = local[0]; t1.p1 = local[2]; t1.p2 = local[3];
    t1.n0 = n; t1.n1 = n; t1.n2 = n;
    t1.has_normals = true;
    t1.uv0 = Point2f(0.f, 0.f); t1.uv1 = Point2f(1.f, 1.f); t1.uv2 = Point2f(0.f, 1.f);
    t1.has_uvs = true;
    localTriangles.push_back(t1);

    return appendMeshGeometry(meshName, materialId, emitterId, transform, normalTransform, localTriangles, out);
}

bool appendSphereShape(const std::string& meshName,
                      float radius,
                      const Point3f& center,
                      int materialId,
                      int emitterId,
                      const Matrix4f& transform,
                      const Matrix4f& normalTransform,
                      LoadedScene& out)
{
    const int stacks = 16;
    const int slices = 16;

    std::vector<std::vector<Point3f>> gridP(stacks + 1, std::vector<Point3f>(slices + 1));
    std::vector<std::vector<Vector3f>> gridN(stacks + 1, std::vector<Vector3f>(slices + 1));
    std::vector<std::vector<Point2f>> gridUV(stacks + 1, std::vector<Point2f>(slices + 1));

    for (int i = 0; i <= stacks; ++i) {
        float theta = i * M_PI / stacks;
        float sinTheta = sinf(theta);
        float cosTheta = cosf(theta);

        for (int j = 0; j <= slices; ++j) {
            float phi = j * 2.f * M_PI / slices;
            float sinPhi = sinf(phi);
            float cosPhi = cosf(phi);

            Vector3f localNormal(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
            Point3f localPos = center + localNormal * radius;

            gridP[i][j] = localPos;
            gridN[i][j] = localNormal;
            gridUV[i][j] = Point2f((float)j / slices, (float)i / stacks);
        }
    }

    std::vector<Triangle> localTriangles;
    localTriangles.reserve(stacks * slices * 2);

    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            Point3f p00 = gridP[i][j];
            Point3f p10 = gridP[i+1][j];
            Point3f p01 = gridP[i][j+1];
            Point3f p11 = gridP[i+1][j+1];

            Vector3f n00 = gridN[i][j];
            Vector3f n10 = gridN[i+1][j];
            Vector3f n01 = gridN[i][j+1];
            Vector3f n11 = gridN[i+1][j+1];

            Point2f uv00 = gridUV[i][j];
            Point2f uv10 = gridUV[i+1][j];
            Point2f uv01 = gridUV[i][j+1];
            Point2f uv11 = gridUV[i+1][j+1];

            // Triangle 1
            Triangle t0;
            t0.p0 = p00; t0.p1 = p10; t0.p2 = p01;
            t0.n0 = n00; t0.n1 = n10; t0.n2 = n01;
            t0.has_normals = true;
            t0.uv0 = uv00; t0.uv1 = uv10; t0.uv2 = uv01;
            t0.has_uvs = true;
            localTriangles.push_back(t0);

            // Triangle 2
            Triangle t1;
            t1.p0 = p10; t1.p1 = p11; t1.p2 = p01;
            t1.n0 = n10; t1.n1 = n11; t1.n2 = n01;
            t1.has_normals = true;
            t1.uv0 = uv10; t1.uv1 = uv11; t1.uv2 = uv01;
            t1.has_uvs = true;
            localTriangles.push_back(t1);
        }
    }

    return appendMeshGeometry(meshName, materialId, emitterId, transform, normalTransform, localTriangles, out);
}

bool appendDiskShape(const std::string& meshName,
                    int materialId,
                    int emitterId,
                    const Matrix4f& transform,
                    const Matrix4f& normalTransform,
                    LoadedScene& out)
{
    const int segments = 32;
    const Vector3f n(0.f, 0.f, 1.f);

    std::vector<Triangle> localTriangles;
    localTriangles.reserve(segments);

    const Point3f center(0.f, 0.f, 0.f);

    for (int i = 0; i < segments; ++i) {
        float angle0 = 2.f * M_PI * (float)i / (float)segments;
        float angle1 = 2.f * M_PI * (float)(i + 1) / (float)segments;

        Point3f p0(cosf(angle0), sinf(angle0), 0.f);
        Point3f p1(cosf(angle1), sinf(angle1), 0.f);

        Triangle t;
        t.p0 = center; t.p1 = p0; t.p2 = p1;
        t.n0 = n; t.n1 = n; t.n2 = n;
        t.has_normals = true;
        t.uv0 = Point2f(0.5f, 0.5f);
        t.uv1 = Point2f(0.5f + 0.5f * cosf(angle0), 0.5f + 0.5f * sinf(angle0));
        t.uv2 = Point2f(0.5f + 0.5f * cosf(angle1), 0.5f + 0.5f * sinf(angle1));
        t.has_uvs = true;
        localTriangles.push_back(t);
    }

    return appendMeshGeometry(meshName, materialId, emitterId, transform, normalTransform, localTriangles, out);
}

bool appendCubeShape(const std::string& meshName,
                    int materialId,
                    int emitterId,
                    const Matrix4f& transform,
                    const Matrix4f& normalTransform,
                    LoadedScene& out)
{
    const Vector3f faceNormals[6] = {
        Vector3f( 1.f,  0.f,  0.f), // +X
        Vector3f(-1.f,  0.f,  0.f), // -X
        Vector3f( 0.f,  1.f,  0.f), // +Y
        Vector3f( 0.f, -1.f,  0.f), // -Y
        Vector3f( 0.f,  0.f,  1.f), // +Z
        Vector3f( 0.f,  0.f, -1.f)  // -Z
    };

    const Point3f faceVertices[6][4] = {
        { Point3f(1.f, -1.f, -1.f), Point3f(1.f,  1.f, -1.f), Point3f(1.f,  1.f,  1.f), Point3f(1.f, -1.f,  1.f) },
        { Point3f(-1.f, -1.f, -1.f), Point3f(-1.f, -1.f,  1.f), Point3f(-1.f,  1.f,  1.f), Point3f(-1.f,  1.f, -1.f) },
        { Point3f(-1.f,  1.f, -1.f), Point3f(-1.f,  1.f,  1.f), Point3f(1.f,  1.f,  1.f), Point3f(1.f,  1.f, -1.f) },
        { Point3f(-1.f, -1.f, -1.f), Point3f(1.f, -1.f, -1.f), Point3f(1.f, -1.f,  1.f), Point3f(-1.f, -1.f,  1.f) },
        { Point3f(-1.f, -1.f,  1.f), Point3f(1.f, -1.f,  1.f), Point3f(1.f,  1.f,  1.f), Point3f(-1.f,  1.f,  1.f) },
        { Point3f(-1.f, -1.f, -1.f), Point3f(-1.f,  1.f, -1.f), Point3f(1.f,  1.f, -1.f), Point3f(1.f, -1.f, -1.f) }
    };

    const Point2f faceUVs[4] = {
        Point2f(0.f, 0.f), Point2f(1.f, 0.f), Point2f(1.f, 1.f), Point2f(0.f, 1.f)
    };

    std::vector<Triangle> localTriangles;
    localTriangles.reserve(12);

    for (int f = 0; f < 6; ++f) {
        Point3f p[4];
        for (int k = 0; k < 4; ++k) {
            p[k] = faceVertices[f][k];
        }

        Vector3f n = faceNormals[f];

        Triangle t0;
        t0.p0 = p[0]; t0.p1 = p[1]; t0.p2 = p[2];
        t0.n0 = n; t0.n1 = n; t0.n2 = n;
        t0.has_normals = true;
        t0.uv0 = faceUVs[0]; t0.uv1 = faceUVs[1]; t0.uv2 = faceUVs[2];
        t0.has_uvs = true;
        localTriangles.push_back(t0);

        Triangle t1;
        t1.p0 = p[0]; t1.p1 = p[2]; t1.p2 = p[3];
        t1.n0 = n; t1.n1 = n; t1.n2 = n;
        t1.has_normals = true;
        t1.uv0 = faceUVs[0]; t1.uv1 = faceUVs[2]; t1.uv2 = faceUVs[3];
        t1.has_uvs = true;
        localTriangles.push_back(t1);
    }

    return appendMeshGeometry(meshName, materialId, emitterId, transform, normalTransform, localTriangles, out);
}

} // namespace futaba
