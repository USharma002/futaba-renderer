#include "scene_loader.h"
#include "proplist.h"
#include "material_builders.h"
#include "triangle.cuh"
#include "material.cuh"

#include <pugixml.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <functional>

namespace fs = std::filesystem;

namespace futaba {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float toF(const std::string& s) { return std::stof(s); }

static void fillPropertyList(const pugi::xml_node& node,
                             PropertyList& plist,
                             const std::function<::Vector3f(const std::string&)>& parseVec3Fn)
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
            // Mitsuba uses <rgb>, some loaders use <color>; treat both identically.
            ::Vector3f c = parseVec3Fn(propVal);
            plist.setColor(propName, ::Color3f(c.x, c.y, c.z));
        }
        else if (tag == "point") {
            ::Vector3f p = parseVec3Fn(propVal);
            plist.setPoint(propName, ::Point3f(p.x, p.y, p.z));
        }
        else if (tag == "vector") {
            ::Vector3f v = parseVec3Fn(propVal);
            plist.setVector(propName, ::Vector3f(v.x, v.y, v.z));
        }
    }
}

::Vector3f SceneLoader::parseVec3(const std::string& s) {
    std::string tmp = s;
    for (char& c : tmp) if (c == ',') c = ' ';
    std::istringstream ss(tmp);
    float x, y, z;
    if (!(ss >> x >> y >> z))
        throw std::runtime_error("Failed to parse vec3 from: '" + s + "'");
    return Vector3f(x, y, z);
}

// ---------------------------------------------------------------------------
// OBJ mesh loader (v + f only, fan-triangulation)
// ---------------------------------------------------------------------------
bool SceneLoader::parseMesh(const std::string& baseDir,
                             const std::string& objFilename,
                             int                materialId,
                             const Matrix4f&    transform,
                             LoadedScene&       out,
                             std::string&       errorOut)
{
    fs::path      objPath = fs::path(baseDir) / objFilename;
    std::ifstream file(objPath);
    if (!file.is_open()) {
        errorOut = "Cannot open OBJ file: " + objPath.string();
        return false;
    }

    std::vector<::Point3f> verts;
    std::vector<::Vector3f> norms;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string token;
        ss >> token;

        if (token == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            ::Point3f p(x, y, z);
            p = transform * p;
            verts.push_back(p);
        }
        else if (token == "vn") {
            float nx, ny, nz;
            ss >> nx >> ny >> nz;
            ::Vector3f n(nx, ny, nz);
            n = normalize(transform * n);
            norms.push_back(n);
        }
        else if (token == "f") {
            std::vector<int> v_indices;
            std::vector<int> n_indices;
            std::string part;
            while (ss >> part) {
                int v_idx = std::stoi(part);
                if (v_idx < 0) v_idx = (int)verts.size() + v_idx + 1;
                v_indices.push_back(v_idx - 1);

                auto first_slash = part.find('/');
                if (first_slash != std::string::npos) {
                    auto second_slash = part.find('/', first_slash + 1);
                    if (second_slash != std::string::npos && second_slash + 1 < part.length()) {
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
            
            for (int i = 1; i + 1 < (int)v_indices.size(); ++i) {
                int i0 = v_indices[0], i1 = v_indices[i], i2 = v_indices[i + 1];
                if (i0 < 0 || i1 < 0 || i2 < 0 ||
                    i0 >= (int)verts.size() || i1 >= (int)verts.size() || i2 >= (int)verts.size())
                {
                    errorOut = "OBJ face index out of range in " + objPath.string();
                    return false;
                }
                Triangle tri;
                tri.p0 = verts[i0];
                tri.p1 = verts[i1];
                tri.p2 = verts[i2];
                
                if (n_indices.size() == v_indices.size() &&
                    n_indices[0] >= 0 && n_indices[i] >= 0 && n_indices[i+1] >= 0 &&
                    n_indices[0] < (int)norms.size() && n_indices[i] < (int)norms.size() && n_indices[i+1] < (int)norms.size()) {
                    tri.n0 = norms[n_indices[0]];
                    tri.n1 = norms[n_indices[i]];
                    tri.n2 = norms[n_indices[i+1]];
                    tri.has_normals = true;
                } else {
                    tri.has_normals = false;
                }
                
                tri.material_id = materialId;
                out.triangles.push_back(tri);
            }
        }
        // vt, mtllib, usemtl, s, o, g — ignored
    }

    if (verts.empty()) {
        errorOut = "OBJ file has no vertices: " + objPath.string();
        return false;
    }
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
    ::Vector3f o = parseVec3(originStr);
    ::Vector3f t = parseVec3(targetStr);
    ::Vector3f u = parseVec3(upStr);

    out.camOrigin = ::Point3f(o.x, o.y, o.z);
    out.camTarget = ::Point3f(t.x, t.y, t.z);
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

    pugi::xml_node root = doc.child("scene");
    if (!root) {
        errorOut = "Root element is not <scene>";
        return false;
    }

    std::string baseDir = fs::path(xmlPath).parent_path().string();
    if (baseDir.empty()) baseDir = ".";

    int nextMatId = 0;

    for (pugi::xml_node node : root.children()) {
        const std::string name = node.name();

        if (name == "mesh") {
            try {
                PropertyList meshProps;
                fillPropertyList(node, meshProps,
                    [this](const std::string& s) { return this->parseVec3(s); });

                const std::string objFile = meshProps.getString("filename");

                PropertyList bsdfProps, emitterProps;
                Matrix4f meshTransform; // Identity by default

                for (const pugi::xml_node& child : node.children()) {
                    const std::string cn = child.name();
                    if (cn == "bsdf") {
                        const char* typeAttr = child.attribute("type").value();
                        if (typeAttr && std::string(typeAttr).length() > 0)
                            bsdfProps.setString("type", typeAttr);
                        fillPropertyList(child, bsdfProps,
                            [this](const std::string& s) { return this->parseVec3(s); });
                    } else if (cn == "emitter") {
                        fillPropertyList(child, emitterProps,
                            [this](const std::string& s) { return this->parseVec3(s); });
                    } else if (cn == "transform") {
                        for (const pugi::xml_node& tchild : child.children()) {
                            std::string tname = tchild.name();
                            if (tname == "translate") {
                                Vector3f t = this->parseVec3(tchild.attribute("value").value());
                                meshTransform = Matrix4f::translate(t) * meshTransform;
                            } else if (tname == "scale") {
                                Vector3f s = this->parseVec3(tchild.attribute("value").value());
                                meshTransform = Matrix4f::scale(s) * meshTransform;
                            } else if (tname == "rotate") {
                                Vector3f axis = this->parseVec3(tchild.attribute("axis").value());
                                float angle = toF(tchild.attribute("angle").value());
                                meshTransform = Matrix4f::rotate(axis, angle) * meshTransform;
                            }
                        }
                    }
                }

                out.materials.emplace_back(makeMaterialFromPropertyLists(bsdfProps, emitterProps));
                if (!parseMesh(baseDir, objFile, nextMatId++, meshTransform, out, errorOut))
                    return false;

            } catch (const std::exception& e) {
                errorOut = std::string("Mesh parse error: ") + e.what();
                return false;
            }
        }
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
                    pugi::xml_node lookat = child.child("lookat");
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
        // sampler, integrator, etc. — silently skipped
    }

    if (out.triangles.empty()) {
        errorOut = "Scene contains no geometry.";
        return false;
    }
    return true;
}

} // namespace futaba