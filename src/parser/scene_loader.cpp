#include "scene_loader.h"
#include "proplist.h"
#include "material_builders.h"
#include "emitter_builders.h"
#include "triangle.cuh"
#include "material.cuh"

#include <pugixml.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <functional>
#include <cmath>
#include <array>
#include <unordered_map>
#include <algorithm>

#include <stb_image.h>
#include <tinyexr.h>

namespace fs = std::filesystem;

namespace futaba {

static Vector3f parseVectorOrPoint(const pugi::xml_node& node,
                                   const std::function<Vector3f(const std::string&)>& parseVec3Fn,
                                   const std::function<std::string(const std::string&)>& resolveValue);

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static float toF(const std::string& s) { return std::stof(s); }

static bool parseMatrix4fValue(const std::string& s, Matrix4f& outM) {
    std::string tmp = s;
    for (char& c : tmp) if (c == ',') c = ' ';
    std::istringstream ss(tmp);

    std::array<float, 16> v{};
    for (int i = 0; i < 16; ++i) {
        if (!(ss >> v[i]))
            return false;
    }

    int k = 0;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            outM.m[r][c] = v[k++];
    return true;
}

static Matrix4f inverseTransposeUpper3x3(const Matrix4f& m) {
    const float a = m.m[0][0], b = m.m[0][1], c = m.m[0][2];
    const float d = m.m[1][0], e = m.m[1][1], f = m.m[1][2];
    const float g = m.m[2][0], h = m.m[2][1], i = m.m[2][2];

    const float A = e * i - f * h;
    const float B = -(d * i - f * g);
    const float C = d * h - e * g;
    const float D = -(b * i - c * h);
    const float E = a * i - c * g;
    const float F = -(a * h - b * g);
    const float G = b * f - c * e;
    const float H = -(a * f - c * d);
    const float I = a * e - b * d;

    const float det = a * A + b * B + c * C;
    if (fabsf(det) <= 1e-12f)
        return Matrix4f();

    const float invDet = 1.f / det;
    Matrix4f n;

    // (M^{-1})^T for the 3x3 linear part.
    n.m[0][0] = A * invDet; n.m[0][1] = B * invDet; n.m[0][2] = C * invDet;
    n.m[1][0] = D * invDet; n.m[1][1] = E * invDet; n.m[1][2] = F * invDet;
    n.m[2][0] = G * invDet; n.m[2][1] = H * invDet; n.m[2][2] = I * invDet;

    n.m[0][3] = n.m[1][3] = n.m[2][3] = 0.f;
    n.m[3][0] = n.m[3][1] = n.m[3][2] = 0.f;
    n.m[3][3] = 1.f;
    return n;
}

static void fillPropertyList(const pugi::xml_node& node,
                             PropertyList& plist,
                             const std::function<Vector3f(const std::string&)>& parseVec3Fn,
                             const std::function<std::string(const std::string&)>& resolveValue)
{
    for (const pugi::xml_node& child : node.children()) {
        const std::string tag      = child.name();
        const std::string propName = child.attribute("name").value();

        if (propName.empty()) continue;

        if      (tag == "boolean") {
            const std::string propVal = resolveValue(child.attribute("value").value());
            plist.setBoolean(propName, propVal == "true" || propVal == "1");
        }
        else if (tag == "integer") {
            const std::string propVal = resolveValue(child.attribute("value").value());
            plist.setInteger(propName, std::stoi(propVal));
        }
        else if (tag == "float")   {
            const std::string propVal = resolveValue(child.attribute("value").value());
            plist.setFloat  (propName, toF(propVal));
        }
        else if (tag == "string")  {
            const std::string propVal = resolveValue(child.attribute("value").value());
            plist.setString (propName, propVal);
        }
        else if (tag == "color" || tag == "rgb" || tag == "spectrum") {
            Vector3f c = parseVectorOrPoint(child, parseVec3Fn, resolveValue);
            plist.setColor(propName, Color3f(c.x, c.y, c.z));
        }
        else if (tag == "point") {
            Vector3f p = parseVectorOrPoint(child, parseVec3Fn, resolveValue);
            plist.setPoint(propName, Point3f(p.x, p.y, p.z));
        }
        else if (tag == "vector") {
            Vector3f v = parseVectorOrPoint(child, parseVec3Fn, resolveValue);
            plist.setVector(propName, Vector3f(v.x, v.y, v.z));
        }
        else if (tag == "texture") {
            std::string texType = child.attribute("type").value();
            std::string texFile = resolveValue(child.attribute("filename").value());
            if (texFile.empty()) {
                for (const pugi::xml_node& texChild : child.children()) {
                    if (std::string(texChild.name()) == "string" &&
                        std::string(texChild.attribute("name").value()) == "filename") {
                        texFile = resolveValue(texChild.attribute("value").value());
                    }
                }
            }
            if (!texFile.empty()) {
                plist.setString(propName + "_texture", texFile);
            }

            // For checkerboard textures, extract color0 and color1 and average them
            if (texType == "checkerboard") {
                Color3f color0(0.4f), color1(0.2f); // reasonable defaults
                for (const pugi::xml_node& texChild : child.children()) {
                    std::string cname = texChild.attribute("name").value();
                    std::string ctag  = texChild.name();
                    if ((ctag == "color" || ctag == "rgb" || ctag == "spectrum") && cname == "color0") {
                        Vector3f v = parseVec3Fn(resolveValue(texChild.attribute("value").value()));
                        color0 = Color3f(v.x, v.y, v.z);
                    } else if ((ctag == "color" || ctag == "rgb" || ctag == "spectrum") && cname == "color1") {
                        Vector3f v = parseVec3Fn(resolveValue(texChild.attribute("value").value()));
                        color1 = Color3f(v.x, v.y, v.z);
                    }
                }
                Color3f avg((color0.x + color1.x) * 0.5f,
                            (color0.y + color1.y) * 0.5f,
                            (color0.z + color1.z) * 0.5f);
                plist.setColor(propName, avg);
            } else {
                // Use white so texture color is the sole contributor
                // (GPU multiplies mat.albedo * texture_sample)
                plist.setColor(propName, Color3f(1.0f));
            }
        }
    }
}

static bool flattenBsdfNode(const pugi::xml_node& bsdfNode,
                             PropertyList& bsdfProps,
                             std::vector<std::string>& warnings,
                             const std::function<Vector3f(const std::string&)>& parseVec3Fn,
                             const std::function<std::string(const std::string&)>& resolveValue)
{
    if (!bsdfNode || std::string(bsdfNode.name()) != "bsdf")
        return false;

    const std::string type = bsdfNode.attribute("type").value();
    if (type == "twosided" || type == "bumpmap" || type == "mask" || type == "normalmap") {
        const pugi::xml_node inner = bsdfNode.child("bsdf");
        if (!inner) {
            warnings.push_back("Found <bsdf type='" + type + "'> without nested <bsdf>; using diffuse fallback.");
            bsdfProps.setString("type", "diffuse");
            return true;
        }
        return flattenBsdfNode(inner, bsdfProps, warnings, parseVec3Fn, resolveValue);
    }

    if (type == "roughconductor") {
        bsdfProps.setString("type", "roughconductor");
        fillPropertyList(bsdfNode, bsdfProps, parseVec3Fn, resolveValue);
        return true;
    }

    bsdfProps.setString("type", type.empty() ? "diffuse" : type);
    fillPropertyList(bsdfNode, bsdfProps, parseVec3Fn, resolveValue);
    return true;
}

static bool appendMeshGeometry(const std::string& meshName,
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

static bool appendRectangleShape(const std::string& meshName,
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

static bool loadEnvMapEXR(const std::string& filename,
                          const std::string& baseDir,
                          std::vector<Color3f>& pixels,
                          int& width,
                          int& height,
                          std::string& errorOut)
{
    fs::path path = fs::path(baseDir) / filename;
    float* rgba = nullptr;
    int w = 0, h = 0;
    const char* err = nullptr;

    int ret = LoadEXR(&rgba, &w, &h, path.string().c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        errorOut = "Failed to load envmap EXR: " + path.string();
        if (err) {
            errorOut += " - " + std::string(err);
            FreeEXRErrorMessage(err);
        }
        return false;
    }

    pixels.resize((size_t)w * (size_t)h);
    for (int i = 0; i < w * h; ++i) {
        pixels[i] = Color3f(rgba[4 * i + 0], rgba[4 * i + 1], rgba[4 * i + 2]);
    }
    free(rgba);
    width = w;
    height = h;
    return true;
}

static bool loadEnvMapHDR(const std::string& filename,
                          const std::string& baseDir,
                          std::vector<Color3f>& pixels,
                          int& width,
                          int& height,
                          std::string& errorOut)
{
    fs::path path = fs::path(baseDir) / filename;
    std::ifstream f(path, std::ios::binary);
    bool isEXR = false;
    if (f.is_open()) {
        char magic[4] = {0};
        if (f.read(magic, 4)) {
            if (magic[0] == 0x76 && magic[1] == 0x2f && magic[2] == 0x31 && magic[3] == 0x01) {
                isEXR = true;
            }
        }
        f.close();
    } else {
        errorOut = "Cannot open environment map file: " + path.string();
        return false;
    }

    if (isEXR) {
        return loadEnvMapEXR(filename, baseDir, pixels, width, height, errorOut);
    }

    int w = 0, h = 0, comp = 0;
    float* data = stbi_loadf(path.string().c_str(), &w, &h, &comp, 3);
    if (!data) {
        errorOut = "Failed to load envmap HDR: " + path.string();
        return false;
    }

    pixels.resize((size_t)w * (size_t)h);
    for (int i = 0; i < w * h; ++i) {
        pixels[i] = Color3f(data[3 * i + 0], data[3 * i + 1], data[3 * i + 2]);
    }
    stbi_image_free(data);
    width = w;
    height = h;
    return true;
}

Vector3f SceneLoader::parseVec3(const std::string& s) {
    if (s.find(':') != std::string::npos) {
        std::vector<std::pair<float, float>> spectrum;
        std::string tmp = s;
        for (char& c : tmp) if (c == ',' || c == '\t' || c == '\r' || c == '\n') c = ' ';
        std::istringstream ss(tmp);
        std::string token;
        while (ss >> token) {
            size_t colon = token.find(':');
            if (colon != std::string::npos) {
                try {
                    float wavelength = std::stof(token.substr(0, colon));
                    float value = std::stof(token.substr(colon + 1));
                    spectrum.push_back({wavelength, value});
                } catch (...) {}
            }
        }
        if (spectrum.empty()) {
            return Vector3f(0.f);
        }
        std::sort(spectrum.begin(), spectrum.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        auto evalSpectrum = [&spectrum](float w) -> float {
            if (w <= spectrum.front().first) return spectrum.front().second;
            if (w >= spectrum.back().first) return spectrum.back().second;
            for (size_t i = 0; i + 1 < spectrum.size(); ++i) {
                if (w >= spectrum[i].first && w <= spectrum[i+1].first) {
                    float t = (w - spectrum[i].first) / (spectrum[i+1].first - spectrum[i].first);
                    return spectrum[i].second * (1.f - t) + spectrum[i+1].second * t;
                }
            }
            return 0.f;
        };

        float r = evalSpectrum(600.f);
        float g = evalSpectrum(550.f);
        float b = evalSpectrum(450.f);
        return Vector3f(r, g, b);
    }

    std::string tmp = s;
    for (char& c : tmp) if (c == ',' || c == '\t' || c == '\r' || c == '\n') c = ' ';
    std::istringstream ss(tmp);
    std::vector<float> values;
    float val;
    while (ss >> val) {
        values.push_back(val);
    }

    if (values.empty()) {
        throw std::runtime_error("Failed to parse vec3 from empty string: '" + s + "'");
    }
    if (values.size() == 1) {
        return Vector3f(values[0], values[0], values[0]);
    }
    if (values.size() == 2) {
        return Vector3f(values[0], values[1], 0.f);
    }
    return Vector3f(values[0], values[1], values[2]);
}

static Vector3f parseVectorOrPoint(const pugi::xml_node& node,
                                   const std::function<Vector3f(const std::string&)>& parseVec3Fn,
                                   const std::function<std::string(const std::string&)>& resolveValue)
{
    std::string valStr = resolveValue(node.attribute("value").value());
    if (!valStr.empty()) {
        return parseVec3Fn(valStr);
    }
    float x = 0.f, y = 0.f, z = 0.f;
    std::string xStr = resolveValue(node.attribute("x").value());
    std::string yStr = resolveValue(node.attribute("y").value());
    std::string zStr = resolveValue(node.attribute("z").value());
    if (!xStr.empty()) x = std::stof(xStr);
    if (!yStr.empty()) y = std::stof(yStr);
    if (!zStr.empty()) z = std::stof(zStr);
    return Vector3f(x, y, z);
}

static bool appendSphereShape(const std::string& meshName,
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
    const float PI = 3.1415926535f;

    std::vector<std::vector<Point3f>> gridP(stacks + 1, std::vector<Point3f>(slices + 1));
    std::vector<std::vector<Vector3f>> gridN(stacks + 1, std::vector<Vector3f>(slices + 1));
    std::vector<std::vector<Point2f>> gridUV(stacks + 1, std::vector<Point2f>(slices + 1));

    for (int i = 0; i <= stacks; ++i) {
        float theta = i * PI / stacks;
        float sinTheta = sinf(theta);
        float cosTheta = cosf(theta);

        for (int j = 0; j <= slices; ++j) {
            float phi = j * 2.f * PI / slices;
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

static bool appendDiskShape(const std::string& meshName,
                            int materialId,
                            int emitterId,
                            const Matrix4f& transform,
                            const Matrix4f& normalTransform,
                            LoadedScene& out)
{
    const int segments = 32;
    const float PI = 3.1415926535f;
    const Vector3f n(0.f, 0.f, 1.f);

    std::vector<Triangle> localTriangles;
    localTriangles.reserve(segments);

    const Point3f center(0.f, 0.f, 0.f);

    for (int i = 0; i < segments; ++i) {
        float angle0 = 2.f * PI * (float)i / (float)segments;
        float angle1 = 2.f * PI * (float)(i + 1) / (float)segments;

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

static bool appendCubeShape(const std::string& meshName,
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

// ---------------------------------------------------------------------------
// OBJ mesh loader (v / vn / f, fan-triangulation)
// ---------------------------------------------------------------------------
bool SceneLoader::parseMesh(const std::string& baseDir,
                             const std::string& objFilename,
                             const std::string& meshName,
                             int                materialId,
                             int                emitterId,
                             const Matrix4f&    transform,
                             const Matrix4f&    normalTransform,
                             LoadedScene&       out,
                             std::string&       errorOut)
{
    fs::path      objPath = fs::path(baseDir) / objFilename;
    std::ifstream file(objPath);
    if (!file.is_open()) {
        errorOut = "Cannot open OBJ file: " + objPath.string();
        return false;
    }

    std::vector<Point3f>  verts;
    std::vector<Vector3f> norms;
    std::vector<Point2f>  texcoords;

    const uint32_t meshTriangleStart = (uint32_t)out.triangles.size();
    const int      meshId            = (int)out.meshes.size();

    // Hoist face-index buffers outside the parsing loop to avoid
    // per-face heap allocation on large meshes.
    std::vector<int> v_indices;
    std::vector<int> n_indices;
    std::vector<int> t_indices;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string token;
        ss >> token;

        if (token == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            Point3f p(x, y, z);
            p = transform * p;          // bake world transform at load time
            verts.push_back(p);
        }
        else if (token == "vn") {
            float nx, ny, nz;
            ss >> nx >> ny >> nz;
            Vector3f n(nx, ny, nz);
            // Normals must use the inverse-transpose of the upper-left 3×3
            // of the transform so they remain perpendicular to the surface
            // under non-uniform scaling.  normalTransform encodes exactly
            // that matrix (built in the caller for the supported ops).
            n = normalize(normalTransform * n);
            norms.push_back(n);
        }
        else if (token == "vt") {
            float u, v;
            ss >> u >> v;
            texcoords.push_back(Point2f(u, v));
        }
        else if (token == "f") {
            v_indices.clear();
            n_indices.clear();
            t_indices.clear();

            std::string part;
            while (ss >> part) {
                int v_idx = 0, vt_idx = 0, vn_idx = 0;
                bool has_vt = false, has_vn = false;

                size_t slash1 = part.find('/');
                if (slash1 == std::string::npos) {
                    v_idx = std::stoi(part);
                } else {
                    v_idx = std::stoi(part.substr(0, slash1));
                    size_t slash2 = part.find('/', slash1 + 1);
                    if (slash2 == std::string::npos) {
                        std::string vt_str = part.substr(slash1 + 1);
                        if (!vt_str.empty()) {
                            vt_idx = std::stoi(vt_str);
                            has_vt = true;
                        }
                    } else {
                        std::string vt_str = part.substr(slash1 + 1, slash2 - slash1 - 1);
                        if (!vt_str.empty()) {
                            vt_idx = std::stoi(vt_str);
                            has_vt = true;
                        }
                        std::string vn_str = part.substr(slash2 + 1);
                        if (!vn_str.empty()) {
                            vn_idx = std::stoi(vn_str);
                            has_vn = true;
                        }
                    }
                }

                if (v_idx < 0) v_idx = (int)verts.size() + v_idx + 1;
                v_indices.push_back(v_idx - 1);

                if (has_vt) {
                    if (vt_idx < 0) vt_idx = (int)texcoords.size() + vt_idx + 1;
                    t_indices.push_back(vt_idx - 1);
                } else {
                    t_indices.push_back(-1);
                }

                if (has_vn) {
                    if (vn_idx < 0) vn_idx = (int)norms.size() + vn_idx + 1;
                    n_indices.push_back(vn_idx - 1);
                } else {
                    n_indices.push_back(-1);
                }
            }

            // Fan-triangulate polygons.
            for (int i = 1; i + 1 < (int)v_indices.size(); ++i) {
                const int i0 = v_indices[0], i1 = v_indices[i], i2 = v_indices[i + 1];
                if (i0 < 0 || i1 < 0 || i2 < 0 ||
                    i0 >= (int)verts.size() ||
                    i1 >= (int)verts.size() ||
                    i2 >= (int)verts.size())
                {
                    errorOut = "OBJ face index out of range in " + objPath.string();
                    return false;
                }

                Triangle tri;
                tri.p0 = verts[i0];
                tri.p1 = verts[i1];
                tri.p2 = verts[i2];

                // Use per-vertex normals when all three are valid.
                if ((int)n_indices.size() == (int)v_indices.size() &&
                    n_indices[0]   >= 0 && n_indices[i]   >= 0 && n_indices[i+1] >= 0 &&
                    n_indices[0]   < (int)norms.size() &&
                    n_indices[i]   < (int)norms.size() &&
                    n_indices[i+1] < (int)norms.size())
                {
                    tri.n0 = norms[n_indices[0]];
                    tri.n1 = norms[n_indices[i]];
                    tri.n2 = norms[n_indices[i+1]];
                    tri.has_normals = true;
                } else {
                    tri.has_normals = false;
                }

                // Use per-vertex texture coords when all three are valid.
                if ((int)t_indices.size() == (int)v_indices.size() &&
                    t_indices[0]   >= 0 && t_indices[i]   >= 0 && t_indices[i+1] >= 0 &&
                    t_indices[0]   < (int)texcoords.size() &&
                    t_indices[i]   < (int)texcoords.size() &&
                    t_indices[i+1] < (int)texcoords.size())
                {
                    tri.uv0 = texcoords[t_indices[0]];
                    tri.uv1 = texcoords[t_indices[i]];
                    tri.uv2 = texcoords[t_indices[i+1]];
                    tri.has_uvs = true;
                } else {
                    tri.has_uvs = false;
                }

                tri.material_id = materialId;
                tri.mesh_id     = meshId;
                out.triangles.push_back(tri);
            }
        }
        // vt, mtllib, usemtl, s, o, g - ignored for now (UV support pending)
    }

    if (verts.empty()) {
        errorOut = "OBJ file has no vertices: " + objPath.string();
        return false;
    }

    // Build the MeshInstance record.
    const uint32_t meshTriangleCount = (uint32_t)out.triangles.size() - meshTriangleStart;

    MeshInstance meshInst;
    meshInst.name          = meshName;
    meshInst.materialId    = materialId;
    meshInst.triangleStart = meshTriangleStart;
    meshInst.triangleCount = meshTriangleCount;
    meshInst.transform     = transform;
    meshInst.emitterType   = (emitterId >= 0) ? EmitterType::Area : EmitterType::None;
    meshInst.emitterId     = emitterId;

    // Compute world-space AABB from the (already-transformed) vertex list.
    meshInst.boundingBoxMin = verts[0];
    meshInst.boundingBoxMax = verts[0];
    for (const auto& v : verts) {
        meshInst.boundingBoxMin.x = std::min(meshInst.boundingBoxMin.x, v.x);
        meshInst.boundingBoxMin.y = std::min(meshInst.boundingBoxMin.y, v.y);
        meshInst.boundingBoxMin.z = std::min(meshInst.boundingBoxMin.z, v.z);
        meshInst.boundingBoxMax.x = std::max(meshInst.boundingBoxMax.x, v.x);
        meshInst.boundingBoxMax.y = std::max(meshInst.boundingBoxMax.y, v.y);
        meshInst.boundingBoxMax.z = std::max(meshInst.boundingBoxMax.z, v.z);
    }

    out.meshes.push_back(meshInst);
    return true;
}

static size_t getPLYTypeSize(const std::string& type) {
    if (type == "char" || type == "uchar" || type == "int8" || type == "uint8") return 1;
    if (type == "short" || type == "ushort" || type == "int16" || type == "uint16") return 2;
    if (type == "int" || type == "uint" || type == "int32" || type == "uint32" || type == "float" || type == "float32") return 4;
    if (type == "double" || type == "float64") return 8;
    return 0;
}

static int readBinaryInt(std::ifstream& file, const std::string& type) {
    if (type == "char" || type == "int8" || type == "uchar" || type == "uint8") {
        uint8_t v = 0; file.read(reinterpret_cast<char*>(&v), 1); return static_cast<int>(v);
    }
    if (type == "short" || type == "int16" || type == "ushort" || type == "uint16") {
        int16_t v = 0; file.read(reinterpret_cast<char*>(&v), 2); return static_cast<int>(v);
    }
    if (type == "int" || type == "int32" || type == "uint" || type == "uint32") {
        int32_t v = 0; file.read(reinterpret_cast<char*>(&v), 4); return static_cast<int>(v);
    }
    return 0;
}

struct PLYProperty {
    std::string name;
    std::string type;
    size_t size = 0;
    size_t offset = 0;
};

struct PLYElement {
    std::string name;
    size_t count = 0;
    std::vector<PLYProperty> properties;
    size_t size = 0;
};

bool SceneLoader::parseMeshPLY(const std::string& baseDir,
                               const std::string& plyFilename,
                               const std::string& meshName,
                               int                materialId,
                               int                emitterId,
                               const Matrix4f&    transform,
                               const Matrix4f&    normalTransform,
                               LoadedScene&       out,
                               std::string&       errorOut)
{
    fs::path plyPath = fs::path(baseDir) / plyFilename;
    std::ifstream file(plyPath, std::ios::binary);
    if (!file.is_open()) {
        errorOut = "Cannot open PLY file: " + plyPath.string();
        return false;
    }

    std::string format;
    std::vector<PLYElement> elements;
    PLYElement* currentElement = nullptr;

    std::string line;
    while (std::getline(file, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' '))
            line.pop_back();
        if (line.empty()) continue;

        std::istringstream ss(line);
        std::string token;
        ss >> token;

        if (token == "ply") continue;
        if (token == "comment") continue;

        if (token == "format") {
            ss >> format;
            continue;
        }

        if (token == "element") {
            std::string elName;
            size_t elCount;
            ss >> elName >> elCount;
            elements.push_back({elName, elCount, {}, 0});
            currentElement = &elements.back();
            continue;
        }

        if (token == "property") {
            if (!currentElement) {
                errorOut = "Property declared before element in PLY: " + line;
                return false;
            }
            std::string propType;
            ss >> propType;
            if (propType == "list") {
                std::string countType, indexType, propName;
                ss >> countType >> indexType >> propName;
                currentElement->properties.push_back({propName, "list:" + countType + ":" + indexType, 0, 0});
            } else {
                std::string propName;
                ss >> propName;
                size_t pSize = getPLYTypeSize(propType);
                currentElement->properties.push_back({propName, propType, pSize, currentElement->size});
                currentElement->size += pSize;
            }
            continue;
        }

        if (token == "end_header") {
            break;
        }
    }

    const PLYElement* vertexElement = nullptr;
    const PLYElement* faceElement = nullptr;
    for (const auto& el : elements) {
        if (el.name == "vertex") vertexElement = &el;
        else if (el.name == "face") faceElement = &el;
    }

    if (!vertexElement || vertexElement->count == 0) {
        errorOut = "PLY file has no vertex element or count is 0.";
        return false;
    }

    int xIdx = -1, yIdx = -1, zIdx = -1;
    int nxIdx = -1, nyIdx = -1, nzIdx = -1;
    int uIdx = -1, vIdx = -1;

    for (int i = 0; i < (int)vertexElement->properties.size(); ++i) {
        const auto& prop = vertexElement->properties[i];
        if (prop.name == "x") xIdx = i;
        else if (prop.name == "y") yIdx = i;
        else if (prop.name == "z") zIdx = i;
        else if (prop.name == "nx") nxIdx = i;
        else if (prop.name == "ny") nyIdx = i;
        else if (prop.name == "nz") nzIdx = i;
        else if (prop.name == "u" || prop.name == "s" || prop.name == "texture_u") uIdx = i;
        else if (prop.name == "v" || prop.name == "t" || prop.name == "texture_v") vIdx = i;
    }

    if (xIdx < 0 || yIdx < 0 || zIdx < 0) {
        errorOut = "PLY vertex element missing x, y, or z coordinates.";
        return false;
    }

    bool hasNormals = (nxIdx >= 0 && nyIdx >= 0 && nzIdx >= 0);
    bool hasUVs = (uIdx >= 0 && vIdx >= 0);

    std::vector<Point3f> verts(vertexElement->count);
    std::vector<Vector3f> norms(hasNormals ? vertexElement->count : 0);
    std::vector<Point2f> texcoords(hasUVs ? vertexElement->count : 0);

    if (format == "binary_little_endian" || format == "binary_little_endian 1.0") {
        std::vector<char> vertexBuffer(vertexElement->count * vertexElement->size);
        file.read(vertexBuffer.data(), vertexBuffer.size());
        if (!file) {
            errorOut = "Failed to read binary vertex data in " + plyPath.string();
            return false;
        }

        const char* ptr = vertexBuffer.data();
        for (size_t i = 0; i < vertexElement->count; ++i) {
            const char* vPtr = ptr + i * vertexElement->size;
            
            auto getVal = [&](int idx) -> float {
                const auto& prop = vertexElement->properties[idx];
                const char* pVal = vPtr + prop.offset;
                if (prop.type == "float" || prop.type == "float32") {
                    float v; std::memcpy(&v, pVal, 4); return v;
                }
                if (prop.type == "double" || prop.type == "float64") {
                    double v; std::memcpy(&v, pVal, 8); return static_cast<float>(v);
                }
                if (prop.type == "int" || prop.type == "int32") {
                    int32_t v; std::memcpy(&v, pVal, 4); return static_cast<float>(v);
                }
                if (prop.type == "uint" || prop.type == "uint32") {
                    uint32_t v; std::memcpy(&v, pVal, 4); return static_cast<float>(v);
                }
                if (prop.type == "uchar" || prop.type == "uint8") {
                    return static_cast<float>(static_cast<uint8_t>(*pVal));
                }
                if (prop.type == "char" || prop.type == "int8") {
                    return static_cast<float>(static_cast<int8_t>(*pVal));
                }
                if (prop.type == "short" || prop.type == "int16") {
                    int16_t v; std::memcpy(&v, pVal, 2); return static_cast<float>(v);
                }
                if (prop.type == "ushort" || prop.type == "uint16") {
                    uint16_t v; std::memcpy(&v, pVal, 2); return static_cast<float>(v);
                }
                return 0.f;
            };

            Point3f p(getVal(xIdx), getVal(yIdx), getVal(zIdx));
            p = transform * p;
            verts[i] = p;

            if (hasNormals) {
                Vector3f n(getVal(nxIdx), getVal(nyIdx), getVal(nzIdx));
                n = normalize(normalTransform * n);
                norms[i] = n;
            }

            if (hasUVs) {
                texcoords[i] = Point2f(getVal(uIdx), getVal(vIdx));
            }
        }
    } else if (format == "ascii" || format == "ascii 1.0") {
        for (size_t i = 0; i < vertexElement->count; ++i) {
            std::string vLine;
            if (!std::getline(file, vLine)) {
                errorOut = "Unexpected end of file while reading ASCII vertices.";
                return false;
            }
            std::istringstream vss(vLine);
            std::vector<float> values;
            float val;
            while (vss >> val) values.push_back(val);
            if (values.size() < vertexElement->properties.size()) {
                errorOut = "Too few values in PLY vertex line: " + vLine;
                return false;
            }

            Point3f p(values[xIdx], values[yIdx], values[zIdx]);
            p = transform * p;
            verts[i] = p;

            if (hasNormals) {
                Vector3f n(values[nxIdx], values[nyIdx], values[nzIdx]);
                n = normalize(normalTransform * n);
                norms[i] = n;
            }

            if (hasUVs) {
                texcoords[i] = Point2f(values[uIdx], values[vIdx]);
            }
        }
    } else {
        errorOut = "Unsupported PLY format: " + format;
        return false;
    }

    if (!faceElement || faceElement->count == 0) {
        errorOut = "PLY file has no face element or count is 0.";
        return false;
    }

    if (faceElement->properties.empty()) {
        errorOut = "PLY face element has no properties.";
        return false;
    }

    const auto& faceProp = faceElement->properties[0];
    if (faceProp.type.rfind("list:", 0) != 0) {
        errorOut = "PLY face property is not a list type: " + faceProp.type;
        return false;
    }

    std::string countType, indexType;
    {
        std::istringstream lss(faceProp.type);
        std::string dummy, cT, iT;
        std::getline(lss, dummy, ':');
        std::getline(lss, cT, ':');
        std::getline(lss, iT, ':');
        countType = cT;
        indexType = iT;
    }

    const uint32_t meshTriangleStart = (uint32_t)out.triangles.size();
    const int meshId = (int)out.meshes.size();

    if (format == "binary_little_endian" || format == "binary_little_endian 1.0") {
        for (size_t f = 0; f < faceElement->count; ++f) {
            int count = readBinaryInt(file, countType);
            if (count < 3) {
                errorOut = "Face has fewer than 3 vertices.";
                return false;
            }
            std::vector<int> faceIndices(count);
            for (int i = 0; i < count; ++i) {
                faceIndices[i] = readBinaryInt(file, indexType);
            }

            for (int i = 1; i + 1 < count; ++i) {
                int i0 = faceIndices[0];
                int i1 = faceIndices[i];
                int i2 = faceIndices[i + 1];

                if (i0 < 0 || i0 >= (int)verts.size() ||
                    i1 < 0 || i1 >= (int)verts.size() ||
                    i2 < 0 || i2 >= (int)verts.size()) {
                    errorOut = "PLY face vertex index out of bounds.";
                    return false;
                }

                Triangle tri;
                tri.p0 = verts[i0];
                tri.p1 = verts[i1];
                tri.p2 = verts[i2];

                if (hasNormals) {
                    tri.n0 = norms[i0];
                    tri.n1 = norms[i1];
                    tri.n2 = norms[i2];
                    tri.has_normals = true;
                } else {
                    tri.has_normals = false;
                }

                if (hasUVs) {
                    tri.uv0 = texcoords[i0];
                    tri.uv1 = texcoords[i1];
                    tri.uv2 = texcoords[i2];
                    tri.has_uvs = true;
                } else {
                    tri.has_uvs = false;
                }

                tri.material_id = materialId;
                tri.mesh_id = meshId;
                out.triangles.push_back(tri);
            }
        }
    } else {
        for (size_t f = 0; f < faceElement->count; ++f) {
            std::string fLine;
            if (!std::getline(file, fLine)) {
                errorOut = "Unexpected end of file while reading ASCII faces.";
                return false;
            }
            std::istringstream fss(fLine);
            int count;
            if (!(fss >> count)) {
                errorOut = "Failed to read face vertex count.";
                return false;
            }
            if (count < 3) {
                errorOut = "Face has fewer than 3 vertices.";
                return false;
            }
            std::vector<int> faceIndices(count);
            for (int i = 0; i < count; ++i) {
                fss >> faceIndices[i];
            }

            for (int i = 1; i + 1 < count; ++i) {
                int i0 = faceIndices[0];
                int i1 = faceIndices[i];
                int i2 = faceIndices[i + 1];

                if (i0 < 0 || i0 >= (int)verts.size() ||
                    i1 < 0 || i1 >= (int)verts.size() ||
                    i2 < 0 || i2 >= (int)verts.size()) {
                    errorOut = "PLY face vertex index out of bounds.";
                    return false;
                }

                Triangle tri;
                tri.p0 = verts[i0];
                tri.p1 = verts[i1];
                tri.p2 = verts[i2];

                if (hasNormals) {
                    tri.n0 = norms[i0];
                    tri.n1 = norms[i1];
                    tri.n2 = norms[i2];
                    tri.has_normals = true;
                } else {
                    tri.has_normals = false;
                }

                if (hasUVs) {
                    tri.uv0 = texcoords[i0];
                    tri.uv1 = texcoords[i1];
                    tri.uv2 = texcoords[i2];
                    tri.has_uvs = true;
                } else {
                    tri.has_uvs = false;
                }

                tri.material_id = materialId;
                tri.mesh_id = meshId;
                out.triangles.push_back(tri);
            }
        }
    }

    const uint32_t meshTriangleCount = (uint32_t)out.triangles.size() - meshTriangleStart;

    MeshInstance meshInst;
    meshInst.name          = meshName;
    meshInst.materialId    = materialId;
    meshInst.triangleStart = meshTriangleStart;
    meshInst.triangleCount = meshTriangleCount;
    meshInst.transform     = transform;
    meshInst.emitterType   = (emitterId >= 0) ? EmitterType::Area : EmitterType::None;
    meshInst.emitterId     = emitterId;

    meshInst.boundingBoxMin = verts[0];
    meshInst.boundingBoxMax = verts[0];
    for (const auto& v : verts) {
        meshInst.boundingBoxMin.x = std::min(meshInst.boundingBoxMin.x, v.x);
        meshInst.boundingBoxMin.y = std::min(meshInst.boundingBoxMin.y, v.y);
        meshInst.boundingBoxMin.z = std::min(meshInst.boundingBoxMin.z, v.z);
        meshInst.boundingBoxMax.x = std::max(meshInst.boundingBoxMax.x, v.x);
        meshInst.boundingBoxMax.y = std::max(meshInst.boundingBoxMax.y, v.y);
        meshInst.boundingBoxMax.z = std::max(meshInst.boundingBoxMax.z, v.z);
    }

    out.meshes.push_back(meshInst);
    return true;
}

// ---------------------------------------------------------------------------
// Camera parser
// ---------------------------------------------------------------------------
bool SceneLoader::parseCamera(const std::string& originStr,
                               const std::string& targetStr,
                               const std::string& upStr,
                               float              fov,
                               LoadedScene&       out,
                               std::string&       /*errorOut*/)
{
    const Vector3f o = parseVec3(originStr);
    const Vector3f t = parseVec3(targetStr);
    const Vector3f u = parseVec3(upStr);

    out.camOrigin = Point3f(o.x, o.y, o.z);
    out.camTarget = Point3f(t.x, t.y, t.z);
    out.camUp     = u;
    out.camFov    = fov;
    out.hasCamera = true;
    return true;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
bool SceneLoader::load(const std::string& xmlPath,
                        LoadedScene&       out,
                        std::string&       errorOut)
{
    out = LoadedScene();

    pugi::xml_document     doc;
    pugi::xml_parse_result result = doc.load_file(xmlPath.c_str());
    if (!result) {
        errorOut = std::string("XML parse error: ") + result.description();
        return false;
    }

    const pugi::xml_node root = doc.child("scene");
    if (!root) {
        errorOut = "Root element is not <scene>";
        return false;
    }

    std::string baseDir = fs::path(xmlPath).parent_path().string();
    if (baseDir.empty()) baseDir = ".";

    int nextMatId = 0;
    std::unordered_map<std::string, int> bsdfIdToMaterial;
    std::unordered_map<std::string, std::string> defaults;

    for (const pugi::xml_node& node : root.children()) {
        if (std::string(node.name()) != "default")
            continue;
        const std::string key = node.attribute("name").value();
        const std::string val = node.attribute("value").value();
        if (!key.empty())
            defaults[key] = val;
    }

    const auto resolveValue = [&defaults](const std::string& raw) -> std::string {
        if (raw.size() > 1 && raw[0] == '$') {
            const std::string key = raw.substr(1);
            const auto it = defaults.find(key);
            if (it != defaults.end())
                return it->second;
        }
        return raw;
    };

    // First pass: collect top-level BSDF definitions so <ref id="..."/> on shapes can resolve.
    for (const pugi::xml_node& node : root.children()) {
        if (std::string(node.name()) != "bsdf")
            continue;

        std::string bsdfId = node.attribute("id").value();
        if (bsdfId.empty()) {
            pugi::xml_node inner = node.child("bsdf");
            while (inner && bsdfId.empty()) {
                bsdfId = inner.attribute("id").value();
                inner = inner.child("bsdf");
            }
        }
        if (bsdfId.empty())
            continue;

        PropertyList bsdfProps;
        if (!flattenBsdfNode(node, bsdfProps, out.warnings,
            [this](const std::string& s) { return this->parseVec3(s); },
            resolveValue))
            continue;

        const int materialId = (int)out.materials.size();
        out.materials.emplace_back(makeMaterialFromPropertyLists(bsdfProps, PropertyList(), out.warnings));
        bsdfIdToMaterial[bsdfId] = materialId;

        std::string texPath;
        if (bsdfProps.hasProperty("reflectance_texture")) texPath = bsdfProps.getString("reflectance_texture");
        else if (bsdfProps.hasProperty("diffuse_reflectance_texture")) texPath = bsdfProps.getString("diffuse_reflectance_texture");
        else if (bsdfProps.hasProperty("specular_reflectance_texture")) texPath = bsdfProps.getString("specular_reflectance_texture");
        out.materialTexturePaths.push_back(texPath);
    }

    for (const pugi::xml_node& node : root.children()) {
        const std::string name = node.name();

        // ----------------------------------------------------------------
        // <mesh> (legacy) and <shape> (Mitsuba-style)
        // ----------------------------------------------------------------
        if (name == "mesh" || name == "shape") {
            try {
                PropertyList meshProps;
                fillPropertyList(node, meshProps,
                    [this](const std::string& s) { return this->parseVec3(s); },
                    resolveValue);

                const std::string shapeType = (name == "shape")
                    ? std::string(node.attribute("type").value())
                    : std::string("obj");

                std::string objFile;
                if (shapeType == "obj" || shapeType == "ply" || name == "mesh")
                    objFile = meshProps.getString("filename");

                std::string meshName = node.attribute("id").value();
                if (meshName.empty()) {
                    meshName = (objFile.empty() ? shapeType : fs::path(objFile).stem().string());
                    if (meshName.empty())
                        meshName = "mesh_" + std::to_string(nextMatId);
                }

                PropertyList bsdfProps, emitterProps;
                int materialId = -1;
                int emitterId = -1;

                bool hasMedium = false;
                Color3f mediumSigmaT(1.f);
                Color3f mediumAlbedo(1.f);
                float mediumScale = 1.f;
                float mediumG = 0.f;

                // Start with identity transforms.
                Matrix4f meshTransform;    // forward transform (for positions)
                Matrix4f normalTransform;  // inverse-transpose (for normals)

                for (const pugi::xml_node& child : node.children()) {
                    const std::string cn = child.name();

                    if (cn == "bsdf") {
                        flattenBsdfNode(child, bsdfProps, out.warnings,
                            [this](const std::string& s) { return this->parseVec3(s); },
                            resolveValue);
                    }
                    else if (cn == "ref") {
                        const std::string refId = child.attribute("id").value();
                        auto it = bsdfIdToMaterial.find(refId);
                        if (it != bsdfIdToMaterial.end()) {
                            materialId = it->second;
                        } else {
                            out.warnings.push_back("Shape references unknown BSDF id '" + refId + "'; using diffuse fallback.");
                        }
                    }
                    else if (cn == "emitter") {
                        const std::string emitterType = child.attribute("type").value();
                        fillPropertyList(child, emitterProps,
                            [this](const std::string& s) { return this->parseVec3(s); },
                            resolveValue);

                        EmitterInstance inst = makeEmitterFromPropertyLists(
                            emitterType, emitterProps, out.warnings);
                        if (inst.type != EmitterType::None) {
                            emitterId = (int)out.emitters.size();
                            out.emitters.push_back(inst);
                        }
                    }
                    else if (cn == "medium") {
                        const std::string mediumType = child.attribute("type").value();
                        if (mediumType == "homogeneous") {
                            hasMedium = true;
                            PropertyList medProps;
                            fillPropertyList(child, medProps,
                                [this](const std::string& s) { return this->parseVec3(s); },
                                resolveValue);
                            mediumSigmaT = medProps.getColor("sigma_t", Color3f(1.f));
                            mediumAlbedo = medProps.getColor("albedo", Color3f(1.f));
                            mediumScale = medProps.getFloat("scale", 1.f);
                            mediumG = medProps.getFloat("g", 0.f);
                            for (const pugi::xml_node& mchild : child.children()) {
                                if (std::string(mchild.name()) == "phase") {
                                    PropertyList phaseProps;
                                    fillPropertyList(mchild, phaseProps,
                                        [this](const std::string& s) { return this->parseVec3(s); },
                                        resolveValue);
                                    mediumG = phaseProps.getFloat("g", 0.f);
                                }
                            }
                        }
                    }
                    else if (cn == "boolean") {
                        const std::string propName = child.attribute("name").value();
                        if (propName == "face_normals") {
                            const bool faceNormals = std::string(child.attribute("value").value()) == "true"
                                                  || std::string(child.attribute("value").value()) == "1";
                            if (faceNormals) {
                                // face_normals=true means flat shading. Current renderer uses a global toggle.
                                // Keep global behavior and warn once per shape to avoid silent mismatch.
                                out.warnings.push_back("Per-shape 'face_normals' requested; renderer currently uses global normal interpolation toggle.");
                            }
                        }
                    }                    else if (cn == "transform") {
                        for (const pugi::xml_node& tchild : child.children()) {
                            const std::string tname = tchild.name();

                            if (tname == "translate") {
                                const Vector3f t = parseVectorOrPoint(tchild, [this](const std::string& s) { return this->parseVec3(s); }, resolveValue);
                                meshTransform = Matrix4f::translate(t) * meshTransform;
                                // Translations do not affect normals - normalTransform unchanged.
                            }
                            else if (tname == "scale") {
                                const Vector3f s = parseVectorOrPoint(tchild, [this](const std::string& s) { return this->parseVec3(s); }, resolveValue);
                                meshTransform = Matrix4f::scale(s) * meshTransform;
                                // Normal scale = (M^{-1})^T = reciprocal scale.
                                // Guard against zero components.
                                const Vector3f invS(
                                    std::abs(s.x) > 1e-9f ? 1.f / s.x : 1.f,
                                    std::abs(s.y) > 1e-9f ? 1.f / s.y : 1.f,
                                    std::abs(s.z) > 1e-9f ? 1.f / s.z : 1.f
                                );
                                normalTransform = Matrix4f::scale(invS) * normalTransform;
                            }
                            else if (tname == "rotate") {
                                Vector3f axis(0.f);
                                std::string axisStr = resolveValue(tchild.attribute("axis").value());
                                if (!axisStr.empty()) {
                                    axis = parseVec3(axisStr);
                                } else {
                                    std::string xStr = resolveValue(tchild.attribute("x").value());
                                    std::string yStr = resolveValue(tchild.attribute("y").value());
                                    std::string zStr = resolveValue(tchild.attribute("z").value());
                                    if (xStr == "1" || xStr == "true") axis.x = 1.f;
                                    if (yStr == "1" || yStr == "true") axis.y = 1.f;
                                    if (zStr == "1" || zStr == "true") axis.z = 1.f;
                                }
                                float len = length(axis);
                                if (len > 1e-6f) {
                                    axis = axis / len;
                                } else {
                                    axis = Vector3f(1.f, 0.f, 0.f);
                                }
                                const float angle = toF(resolveValue(tchild.attribute("angle").value()));
                                meshTransform   = Matrix4f::rotate(axis, angle) * meshTransform;
                                // For rotations R^{-T} = R (orthogonal matrix).
                                normalTransform = Matrix4f::rotate(axis, angle) * normalTransform;
                            }
                            else if (tname == "matrix") {
                                Matrix4f explicitM;
                                if (parseMatrix4fValue(resolveValue(tchild.attribute("value").value()), explicitM)) {
                                    meshTransform = explicitM;
                                    normalTransform = inverseTransposeUpper3x3(explicitM);
                                }
                            }
                        }
                    }
                }

                if (materialId < 0) {
                    out.materials.emplace_back(
                        makeMaterialFromPropertyLists(bsdfProps, emitterProps, out.warnings));
                    materialId = (int)out.materials.size() - 1;

                    std::string texPath;
                    if (bsdfProps.hasProperty("reflectance_texture")) texPath = bsdfProps.getString("reflectance_texture");
                    else if (bsdfProps.hasProperty("diffuse_reflectance_texture")) texPath = bsdfProps.getString("diffuse_reflectance_texture");
                    else if (bsdfProps.hasProperty("specular_reflectance_texture")) texPath = bsdfProps.getString("specular_reflectance_texture");
                    out.materialTexturePaths.push_back(texPath);
                }

                if (shapeType == "obj" || name == "mesh") {
                    if (!parseMesh(baseDir, objFile, meshName, materialId,
                                   emitterId, meshTransform, normalTransform, out, errorOut))
                        return false;
                } else if (shapeType == "ply") {
                    if (!parseMeshPLY(baseDir, objFile, meshName, materialId,
                                      emitterId, meshTransform, normalTransform, out, errorOut))
                        return false;
                } else if (shapeType == "rectangle") {
                    appendRectangleShape(meshName, materialId, emitterId,
                                         meshTransform, normalTransform, out);
                } else if (shapeType == "sphere") {
                    float radius = meshProps.getFloat("radius", 1.f);
                    Point3f center = meshProps.getPoint("center", Point3f(0.f));
                    appendSphereShape(meshName, radius, center, materialId, emitterId,
                                      meshTransform, normalTransform, out);
                } else if (shapeType == "cube") {
                    appendCubeShape(meshName, materialId, emitterId,
                                    meshTransform, normalTransform, out);
                } else if (shapeType == "disk") {
                    appendDiskShape(meshName, materialId, emitterId,
                                    meshTransform, normalTransform, out);
                } else {
                    out.warnings.push_back("Unsupported shape type '" + shapeType + "'; skipping shape '" + meshName + "'.");
                }

                if (hasMedium) {
                    out.hasMedium = true;
                    out.mediumMeshId = (int)out.meshes.size() - 1;
                    out.mediumSigmaT = mediumSigmaT * mediumScale;
                    out.mediumSigmaS = mediumAlbedo * out.mediumSigmaT;
                    out.mediumSigmaA = out.mediumSigmaT - out.mediumSigmaS;
                    out.mediumG = mediumG;
                }

                ++nextMatId;

            } catch (const std::exception& e) {
                errorOut = std::string("Mesh parse error: ") + e.what();
                return false;
            }
        }

        // ----------------------------------------------------------------
        // <emitter type="envmap"> (Mitsuba-style environment lighting)
        // ----------------------------------------------------------------
        else if (name == "emitter") {
            const std::string emitterType = node.attribute("type").value();
            if (emitterType == "envmap") {
                PropertyList envProps;
                fillPropertyList(node, envProps,
                    [this](const std::string& s) { return this->parseVec3(s); },
                    resolveValue);

                Matrix4f envTransform;
                bool hasTransform = false;
                for (const pugi::xml_node& child : node.children()) {
                    if (std::string(child.name()) != "transform")
                        continue;
                    const pugi::xml_node mnode = child.child("matrix");
                    if (mnode && parseMatrix4fValue(resolveValue(mnode.attribute("value").value()), envTransform)) {
                        hasTransform = true;
                    }
                }

                const std::string filename = envProps.getString("filename", std::string());
                if (filename.empty()) {
                    out.warnings.push_back("envmap emitter missing filename; background will remain black.");
                    continue;
                }

                if (!loadEnvMapHDR(filename, baseDir, out.envMapPixels, out.envMapWidth, out.envMapHeight, errorOut))
                    return false;

                out.hasEnvMap = true;
                out.envMapToWorld = hasTransform ? envTransform : Matrix4f();
            } else if (emitterType == "constant") {
                PropertyList envProps;
                fillPropertyList(node, envProps,
                    [this](const std::string& s) { return this->parseVec3(s); },
                    resolveValue);
                const Color3f radiance = envProps.getColor("radiance",
                                           envProps.getColor("emission", Color3f(0.f)));
                out.hasConstantEnv = true;
                out.constantEnv = radiance;
            }
        }

        // ----------------------------------------------------------------
        // <camera> (legacy) / <sensor type="perspective"> (Mitsuba-style)
        // ----------------------------------------------------------------
        else if (name == "camera" || name == "sensor") {
            PropertyList cameraProps;
            fillPropertyList(node, cameraProps,
                [this](const std::string& s) { return this->parseVec3(s); },
                resolveValue);

            float fov = cameraProps.getFloat("fov", 45.f);
            std::string originStr, targetStr, upStr;
            Matrix4f sensorToWorld;
            bool hasSensorMatrix = false;

            for (const pugi::xml_node& child : node.children()) {
                const std::string cn = child.name();
                if (cn == "float") {
                    if (std::string(child.attribute("name").value()) == "fov")
                        fov = toF(resolveValue(child.attribute("value").value()));
                } else if (cn == "transform") {
                    const pugi::xml_node lookat = child.child("lookat");
                    if (lookat) {
                        originStr = resolveValue(lookat.attribute("origin").value());
                        targetStr = resolveValue(lookat.attribute("target").value());
                        upStr     = resolveValue(lookat.attribute("up").value());
                    } else {
                        const pugi::xml_node mnode = child.child("matrix");
                        if (mnode && parseMatrix4fValue(resolveValue(mnode.attribute("value").value()), sensorToWorld)) {
                            hasSensorMatrix = true;
                        }
                    }
                }
            }

            if (!originStr.empty()) {
                if (!parseCamera(originStr, targetStr, upStr, fov, out, errorOut))
                    return false;
            } else if (hasSensorMatrix) {
                // Mitsuba's sensor transform is a local-to-world transform.
                // Derive the camera frame by transforming the local basis.
                const Point3f camO   = sensorToWorld * Point3f(0.f, 0.f, 0.f);
                const Point3f camFwdP = sensorToWorld * Point3f(0.f, 0.f, 1.f);
                const Vector3f camUpV = sensorToWorld * Vector3f(0.f, 1.f, 0.f);
                const Vector3f camDir = normalize(camFwdP - camO);
                const Vector3f camUp   = normalize(camUpV);

                out.camOrigin = camO;
                out.camTarget = camO + camDir;
                out.camUp     = camUp;
                out.camFov    = fov;
                out.hasCamera = true;
            }
        }
        else if (name == "integrator") {
            out.integratorType = resolveValue(node.attribute("type").value());
        }
    }

    if (out.triangles.empty()) {
        errorOut = "Scene contains no geometry.";
        return false;
    }
    return true;
}

} // namespace futaba