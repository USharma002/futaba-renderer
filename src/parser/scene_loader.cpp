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

namespace fs = std::filesystem;

namespace futaba {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static float toF(const std::string& s) { return std::stof(s); }

static void fillPropertyList(const pugi::xml_node& node,
                             PropertyList& plist,
                             const std::function<Vector3f(const std::string&)>& parseVec3Fn)
{
    for (const pugi::xml_node& child : node.children()) {
        const std::string tag      = child.name();
        const std::string propName = child.attribute("name").value();
        const std::string propVal  = child.attribute("value").value();

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

    for (const pugi::xml_node& node : root.children()) {
        const std::string name = node.name();

        // ----------------------------------------------------------------
        // <mesh>
        // ----------------------------------------------------------------
        if (name == "mesh") {
            try {
                PropertyList meshProps;
                fillPropertyList(node, meshProps,
                    [this](const std::string& s) { return this->parseVec3(s); });

                const std::string objFile = meshProps.getString("filename");

                std::string meshName = node.attribute("id").value();
                if (meshName.empty()) {
                    meshName = fs::path(objFile).stem().string();
                    if (meshName.empty())
                        meshName = "mesh_" + std::to_string(nextMatId);
                }

                PropertyList bsdfProps, emitterProps;
                int emitterId = -1;

                // Start with identity transforms.
                Matrix4f meshTransform;    // forward transform (for positions)
                Matrix4f normalTransform;  // inverse-transpose (for normals)

                for (const pugi::xml_node& child : node.children()) {
                    const std::string cn = child.name();

                    if (cn == "bsdf") {
                        const char* typeAttr = child.attribute("type").value();
                        if (typeAttr && typeAttr[0] != '\0')
                            bsdfProps.setString("type", typeAttr);
                        fillPropertyList(child, bsdfProps,
                            [this](const std::string& s) { return this->parseVec3(s); });
                    }
                    else if (cn == "emitter") {
                        const std::string emitterType = child.attribute("type").value();
                        fillPropertyList(child, emitterProps,
                            [this](const std::string& s) { return this->parseVec3(s); });

                        EmitterInstance inst = makeEmitterFromPropertyLists(
                            emitterType, emitterProps, out.warnings);
                        if (inst.type != EmitterType::None) {
                            emitterId = (int)out.emitters.size();
                            out.emitters.push_back(inst);
                        }
                    }
                    else if (cn == "transform") {
                        for (const pugi::xml_node& tchild : child.children()) {
                            const std::string tname = tchild.name();

                            if (tname == "translate") {
                                const Vector3f t = parseVec3(tchild.attribute("value").value());
                                meshTransform = Matrix4f::translate(t) * meshTransform;
                                // Translations do not affect normals - normalTransform unchanged.
                            }
                            else if (tname == "scale") {
                                const Vector3f s = parseVec3(tchild.attribute("value").value());
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
                                const Vector3f axis  = parseVec3(tchild.attribute("axis").value());
                                const float    angle = toF(tchild.attribute("angle").value());
                                meshTransform   = Matrix4f::rotate(axis, angle) * meshTransform;
                                // For rotations R^{-T} = R (orthogonal matrix).
                                normalTransform = Matrix4f::rotate(axis, angle) * normalTransform;
                            }
                        }
                    }
                }

                out.materials.emplace_back(
                    makeMaterialFromPropertyLists(bsdfProps, emitterProps, out.warnings));

                if (!parseMesh(baseDir, objFile, meshName, nextMatId++,
                               emitterId, meshTransform, normalTransform, out, errorOut))
                    return false;

            } catch (const std::exception& e) {
                errorOut = std::string("Mesh parse error: ") + e.what();
                return false;
            }
        }

        // ----------------------------------------------------------------
        // <camera>
        // ----------------------------------------------------------------
        else if (name == "camera") {
            PropertyList cameraProps;
            fillPropertyList(node, cameraProps,
                [this](const std::string& s) { return this->parseVec3(s); });

            float fov = cameraProps.getFloat("fov", 45.f);
            std::string originStr, targetStr, upStr;

            for (const pugi::xml_node& child : node.children()) {
                const std::string cn = child.name();
                if (cn == "float") {
                    if (std::string(child.attribute("name").value()) == "fov")
                        fov = toF(child.attribute("value").value());
                } else if (cn == "transform") {
                    const pugi::xml_node lookat = child.child("lookat");
                    if (lookat) {
                        originStr = lookat.attribute("origin").value();
                        targetStr = lookat.attribute("target").value();
                        upStr     = lookat.attribute("up").value();
                    }
                }
            }

            if (!originStr.empty()) {
                if (!parseCamera(originStr, targetStr, upStr, fov, out, errorOut))
                    return false;
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