#include "shape_factory.h"
#include "scene_loader.h"
#include "common.cuh"
#include <algorithm>
#include <cmath>

FUTABA_NAMESPACE_BEGIN

bool appendMeshGeometry(const std::string& meshName,
                       int materialId,
                       int emitterId,
                       const Matrix4f& transform,
                       const Matrix4f& normalTransform,
                       const std::vector<Triangle>& localTriangles,
                       CPUScene& out)
{
    const uint32_t meshTriangleStart = (uint32_t)out.triangles.size();
    const int meshId = (int)out.meshes.size();

    AABB bbox;

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
            bbox.grow(p);
        }

        out.triangles.push_back(t);
    }

    MeshInstance meshInst;
    meshInst.materialId    = materialId;
    meshInst.triangleStart = meshTriangleStart;
    meshInst.triangleCount = (uint32_t)localTriangles.size();
    meshInst.transform     = transform;
    meshInst.emitterType   = (emitterId >= 0) ? EmitterType::Area : EmitterType::None;
    meshInst.emitterId     = emitterId;
    meshInst.bbox          = bbox;

    out.meshes.push_back(meshInst);
    out.meshNames.push_back(meshName);
    return true;
}

namespace {
inline void appendQuad(std::vector<Triangle>& tris,
                       const Point3f& p0, const Point3f& p1, const Point3f& p2, const Point3f& p3,
                       const Vector3f& n,
                       const Point2f& uv0 = Point2f(0.f, 0.f), const Point2f& uv1 = Point2f(1.f, 0.f),
                       const Point2f& uv2 = Point2f(1.f, 1.f), const Point2f& uv3 = Point2f(0.f, 1.f))
{
    Triangle t0;
    t0.p0 = p0; t0.p1 = p1; t0.p2 = p2;
    t0.n0 = n; t0.n1 = n; t0.n2 = n;
    t0.uv0 = uv0; t0.uv1 = uv1; t0.uv2 = uv2;
    t0.has_normals = t0.has_uvs = true;

    Triangle t1;
    t1.p0 = p0; t1.p1 = p2; t1.p2 = p3;
    t1.n0 = n; t1.n1 = n; t1.n2 = n;
    t1.uv0 = uv0; t1.uv1 = uv2; t1.uv2 = uv3;
    t1.has_normals = t1.has_uvs = true;

    tris.push_back(t0);
    tris.push_back(t1);
}
} // namespace

bool appendRectangleShape(const std::string& meshName,
                         int materialId,
                         int emitterId,
                         const Matrix4f& transform,
                         const Matrix4f& normalTransform,
                         CPUScene& out)
{
    std::vector<Triangle> localTriangles;
    localTriangles.reserve(2);
    appendQuad(localTriangles,
               Point3f(-1.f, -1.f, 0.f), Point3f( 1.f, -1.f, 0.f),
               Point3f( 1.f,  1.f, 0.f), Point3f(-1.f,  1.f, 0.f),
               Vector3f(0.f, 0.f, 1.f));
    return appendMeshGeometry(meshName, materialId, emitterId, transform, normalTransform, localTriangles, out);
}

bool appendSphereShape(const std::string& meshName,
                      float radius,
                      const Point3f& center,
                      int materialId,
                      int emitterId,
                      const Matrix4f& transform,
                      const Matrix4f& normalTransform,
                      CPUScene& out)
{
    const int stacks = 16, slices = 16;
    Point3f  gridP[stacks + 1][slices + 1];
    Vector3f gridN[stacks + 1][slices + 1];
    Point2f  gridUV[stacks + 1][slices + 1];

    for (int i = 0; i <= stacks; ++i) {
        float theta = i * M_PI / stacks;
        float sinTheta, cosTheta;
        Warp::fast_sincos(theta, &sinTheta, &cosTheta);

        for (int j = 0; j <= slices; ++j) {
            float phi = j * 2.f * M_PI / slices;
            float sinPhi, cosPhi;
            Warp::fast_sincos(phi, &sinPhi, &cosPhi);

            Vector3f localNormal(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
            gridP[i][j] = center + localNormal * radius;
            gridN[i][j] = localNormal;
            gridUV[i][j] = Point2f((float)j / slices, (float)i / stacks);
        }
    }

    std::vector<Triangle> localTriangles;
    localTriangles.reserve(stacks * slices * 2);

    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            Triangle t0;
            t0.p0 = gridP[i][j]; t0.p1 = gridP[i+1][j]; t0.p2 = gridP[i][j+1];
            t0.n0 = gridN[i][j]; t0.n1 = gridN[i+1][j]; t0.n2 = gridN[i][j+1];
            t0.uv0 = gridUV[i][j]; t0.uv1 = gridUV[i+1][j]; t0.uv2 = gridUV[i][j+1];
            t0.has_normals = t0.has_uvs = true;

            Triangle t1;
            t1.p0 = gridP[i+1][j]; t1.p1 = gridP[i+1][j+1]; t1.p2 = gridP[i][j+1];
            t1.n0 = gridN[i+1][j]; t1.n1 = gridN[i+1][j+1]; t1.n2 = gridN[i][j+1];
            t1.uv0 = gridUV[i+1][j]; t1.uv1 = gridUV[i+1][j+1]; t1.uv2 = gridUV[i][j+1];
            t1.has_normals = t1.has_uvs = true;

            localTriangles.push_back(t0);
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
                    CPUScene& out)
{
    const int segments = 32;
    const Vector3f n(0.f, 0.f, 1.f);
    const Point3f center(0.f, 0.f, 0.f);

    std::vector<Triangle> localTriangles;
    localTriangles.reserve(segments);

    for (int i = 0; i < segments; ++i) {
        float a0 = 2.f * M_PI * (float)i / (float)segments;
        float a1 = 2.f * M_PI * (float)(i + 1) / (float)segments;
        float s0, c0, s1, c1;
        Warp::fast_sincos(a0, &s0, &c0);
        Warp::fast_sincos(a1, &s1, &c1);

        Triangle t;
        t.p0 = center; t.p1 = Point3f(c0, s0, 0.f); t.p2 = Point3f(c1, s1, 0.f);
        t.n0 = n; t.n1 = n; t.n2 = n;
        t.has_normals = true;
        t.uv0 = Point2f(0.5f, 0.5f);
        t.uv1 = Point2f(0.5f + 0.5f * c0, 0.5f + 0.5f * s0);
        t.uv2 = Point2f(0.5f + 0.5f * c1, 0.5f + 0.5f * s1);
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
                    CPUScene& out)
{
    const Vector3f faceNormals[6] = {
        Vector3f( 1.f,  0.f,  0.f), Vector3f(-1.f,  0.f,  0.f),
        Vector3f( 0.f,  1.f,  0.f), Vector3f( 0.f, -1.f,  0.f),
        Vector3f( 0.f,  0.f,  1.f), Vector3f( 0.f,  0.f, -1.f)
    };

    const Point3f faceVertices[6][4] = {
        { Point3f( 1.f, -1.f, -1.f), Point3f( 1.f,  1.f, -1.f), Point3f( 1.f,  1.f,  1.f), Point3f( 1.f, -1.f,  1.f) },
        { Point3f(-1.f, -1.f, -1.f), Point3f(-1.f, -1.f,  1.f), Point3f(-1.f,  1.f,  1.f), Point3f(-1.f,  1.f, -1.f) },
        { Point3f(-1.f,  1.f, -1.f), Point3f(-1.f,  1.f,  1.f), Point3f( 1.f,  1.f,  1.f), Point3f( 1.f,  1.f, -1.f) },
        { Point3f(-1.f, -1.f, -1.f), Point3f( 1.f, -1.f, -1.f), Point3f( 1.f, -1.f,  1.f), Point3f(-1.f, -1.f,  1.f) },
        { Point3f(-1.f, -1.f,  1.f), Point3f( 1.f, -1.f,  1.f), Point3f( 1.f,  1.f,  1.f), Point3f(-1.f,  1.f,  1.f) },
        { Point3f(-1.f, -1.f, -1.f), Point3f(-1.f,  1.f, -1.f), Point3f( 1.f,  1.f, -1.f), Point3f( 1.f, -1.f, -1.f) }
    };

    std::vector<Triangle> localTriangles;
    localTriangles.reserve(12);

    for (int f = 0; f < 6; ++f) {
        appendQuad(localTriangles,
                   faceVertices[f][0], faceVertices[f][1], faceVertices[f][2], faceVertices[f][3],
                   faceNormals[f]);
    }

    return appendMeshGeometry(meshName, materialId, emitterId, transform, normalTransform, localTriangles, out);
}

FUTABA_NAMESPACE_END