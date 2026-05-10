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

#include <stb_image.h>

namespace fs = std::filesystem;

namespace futaba {

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
        const std::string propValRaw = child.attribute("value").value();
        const std::string propVal    = resolveValue(propValRaw);

        if (propName.empty()) continue;

        if      (tag == "boolean") { plist.setBoolean(propName, propVal == "true" || propVal == "1"); }
        else if (tag == "integer") { plist.setInteger(propName, std::stoi(propVal)); }
        else if (tag == "float")   { plist.setFloat  (propName, toF(propVal)); }
        else if (tag == "string")  { plist.setString (propName, propVal); }
        else if (tag == "color" || tag == "rgb" || tag == "spectrum") {
            Vector3f c = parseVec3Fn(propVal);
            plist.setColor(propName, Color3f(c.x, c.y, c.z));
        }
        else if (tag == "point") {
            Vector3f p = parseVec3Fn(propVal);
            plist.setPoint(propName, Point3f(p.x, p.y, p.z));
        }
        else if (tag == "vector") {
            Vector3f v = parseVec3Fn(propVal);
            plist.setVector(propName, Vector3f(v.x, v.y, v.z));
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
    if (type == "twosided") {
        const pugi::xml_node inner = bsdfNode.child("bsdf");
        if (!inner) {
            warnings.push_back("Found <bsdf type='twosided'> without nested <bsdf>; using diffuse fallback.");
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

static bool appendRectangleShape(const std::string& meshName,
                                 int materialId,
                                 int emitterId,
                                 const Matrix4f& transform,
                                 const Matrix4f& normalTransform,
                                 LoadedScene& out)
{
    const uint32_t meshTriangleStart = (uint32_t)out.triangles.size();
    const int meshId = (int)out.meshes.size();

    const Point3f local[4] = {
        Point3f(-1.f, -1.f, 0.f),
        Point3f( 1.f, -1.f, 0.f),
        Point3f( 1.f,  1.f, 0.f),
        Point3f(-1.f,  1.f, 0.f)
    };

    Point3f p[4];
    for (int k = 0; k < 4; ++k)
        p[k] = transform * local[k];

    Vector3f n = normalize(normalTransform * Vector3f(0.f, 0.f, 1.f));

    Triangle t0;
    t0.p0 = p[0]; t0.p1 = p[1]; t0.p2 = p[2];
    t0.n0 = n; t0.n1 = n; t0.n2 = n;
    t0.has_normals = true;
    t0.material_id = materialId;
    t0.mesh_id = meshId;
    out.triangles.push_back(t0);

    Triangle t1;
    t1.p0 = p[0]; t1.p1 = p[2]; t1.p2 = p[3];
    t1.n0 = n; t1.n1 = n; t1.n2 = n;
    t1.has_normals = true;
    t1.material_id = materialId;
    t1.mesh_id = meshId;
    out.triangles.push_back(t1);

    MeshInstance meshInst;
    meshInst.name          = meshName;
    meshInst.materialId    = materialId;
    meshInst.triangleStart = meshTriangleStart;
    meshInst.triangleCount = 2;
    meshInst.transform     = transform;
    meshInst.emitterType   = (emitterId >= 0) ? EmitterType::Area : EmitterType::None;
    meshInst.emitterId     = emitterId;

    meshInst.boundingBoxMin = p[0];
    meshInst.boundingBoxMax = p[0];
    for (int k = 1; k < 4; ++k) {
        meshInst.boundingBoxMin.x = std::min(meshInst.boundingBoxMin.x, p[k].x);
        meshInst.boundingBoxMin.y = std::min(meshInst.boundingBoxMin.y, p[k].y);
        meshInst.boundingBoxMin.z = std::min(meshInst.boundingBoxMin.z, p[k].z);
        meshInst.boundingBoxMax.x = std::max(meshInst.boundingBoxMax.x, p[k].x);
        meshInst.boundingBoxMax.y = std::max(meshInst.boundingBoxMax.y, p[k].y);
        meshInst.boundingBoxMax.z = std::max(meshInst.boundingBoxMax.z, p[k].z);
    }

    out.meshes.push_back(meshInst);
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
    std::string tmp = s;
    for (char& c : tmp) if (c == ',') c = ' ';
    std::istringstream ss(tmp);
    float x, y, z;
    if (!(ss >> x >> y >> z))
        throw std::runtime_error("Failed to parse vec3 from: '" + s + "'");
    return Vector3f(x, y, z);
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

    const uint32_t meshTriangleStart = (uint32_t)out.triangles.size();
    const int      meshId            = (int)out.meshes.size();

    // Hoist face-index buffers outside the parsing loop to avoid
    // per-face heap allocation on large meshes.
    std::vector<int> v_indices;
    std::vector<int> n_indices;

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
        else if (token == "f") {
            v_indices.clear();
            n_indices.clear();

            std::string part;
            while (ss >> part) {
                int v_idx = std::stoi(part);
                // OBJ uses 1-based indices; negative values are relative to
                // the end of the current vertex list.
                if (v_idx < 0) v_idx = (int)verts.size() + v_idx + 1;
                v_indices.push_back(v_idx - 1);

                // Parse optional normal index from v[/vt[/vn]] syntax.
                const auto first_slash  = part.find('/');
                if (first_slash != std::string::npos) {
                    const auto second_slash = part.find('/', first_slash + 1);
                    if (second_slash != std::string::npos &&
                        second_slash + 1 < part.size())
                    {
                        int n_idx = std::stoi(part.substr(second_slash + 1));
                        if (n_idx < 0) n_idx = (int)norms.size() + n_idx + 1;
                        n_indices.push_back(n_idx - 1);
                    } else {
                        n_indices.push_back(-1);
                    }
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

        const std::string bsdfId = node.attribute("id").value();
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
                if (shapeType == "obj" || name == "mesh")
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
                    }
                    else if (cn == "transform") {
                        for (const pugi::xml_node& tchild : child.children()) {
                            const std::string tname = tchild.name();

                            if (tname == "translate") {
                                const Vector3f t = parseVec3(resolveValue(tchild.attribute("value").value()));
                                meshTransform = Matrix4f::translate(t) * meshTransform;
                                // Translations do not affect normals - normalTransform unchanged.
                            }
                            else if (tname == "scale") {
                                const Vector3f s = parseVec3(resolveValue(tchild.attribute("value").value()));
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
                                const Vector3f axis  = parseVec3(resolveValue(tchild.attribute("axis").value()));
                                const float    angle = toF(resolveValue(tchild.attribute("angle").value()));
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
                }

                if (shapeType == "obj" || name == "mesh") {
                    if (!parseMesh(baseDir, objFile, meshName, materialId,
                                   emitterId, meshTransform, normalTransform, out, errorOut))
                        return false;
                } else if (shapeType == "rectangle") {
                    appendRectangleShape(meshName, materialId, emitterId,
                                         meshTransform, normalTransform, out);
                } else {
                    out.warnings.push_back("Unsupported shape type '" + shapeType + "'; skipping shape '" + meshName + "'.");
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
        // sampler, integrator, etc. - silently skipped
    }

    if (out.triangles.empty()) {
        errorOut = "Scene contains no geometry.";
        return false;
    }
    return true;
}

} // namespace futaba