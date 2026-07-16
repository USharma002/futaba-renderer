#include "scene_loader.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <happly.h>

#include "proplist.h"
#include "material_builders.h"
#include "emitter_builders.h"
#include "triangle.cuh"
#include "material.cuh"
#include "shape_factory.h"
#include "envmap_loader.h"

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

FUTABA_NAMESPACE_BEGIN

namespace {

enum ETag {
    // Objects
    EScene,
    EMesh,
    EBsdf,
    EEmitter,
    ECamera,
    EIntegrator,
    ERef,
    ETransform,

    // Properties
    EBoolean,
    EInteger,
    EFloat,
    EString,
    EPoint,
    EVector,
    EColor,
    ETexture,

    // Transform Operations
    ETranslate,
    EScale,
    ERotate,
    EMatrix,

    EInvalid
};

static ETag getTag(const std::string& name) {
    static const std::unordered_map<std::string, ETag> tags = {
        { "scene",      EScene },
        { "mesh",       EMesh },
        { "shape",      EMesh },
        { "bsdf",       EBsdf },
        { "emitter",    EEmitter },
        { "camera",     ECamera },
        { "sensor",     ECamera },
        { "integrator", EIntegrator },
        { "ref",        ERef },
        { "transform",  ETransform },
        { "boolean",    EBoolean },
        { "integer",    EInteger },
        { "float",      EFloat },
        { "string",     EString },
        { "point",      EPoint },
        { "vector",     EVector },
        { "color",      EColor },
        { "rgb",        EColor },
        { "spectrum",   EColor },
        { "texture",    ETexture },
        { "translate",  ETranslate },
        { "scale",      EScale },
        { "rotate",     ERotate },
        { "matrix",     EMatrix }
    };
    auto it = tags.find(name);
    return (it != tags.end()) ? it->second : EInvalid;
}

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

    n.m[0][0] = A * invDet; n.m[0][1] = B * invDet; n.m[0][2] = C * invDet;
    n.m[1][0] = D * invDet; n.m[1][1] = E * invDet; n.m[1][2] = F * invDet;
    n.m[2][0] = G * invDet; n.m[2][1] = H * invDet; n.m[2][2] = I * invDet;

    n.m[0][3] = n.m[1][3] = n.m[2][3] = 0.f;
    n.m[3][0] = n.m[3][1] = n.m[3][2] = 0.f;
    n.m[3][3] = 1.f;
    return n;
}

static Vector3f parseVec3(const std::string& s) {
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

static std::string resolveValue(const std::string& raw, const std::unordered_map<std::string, std::string>& defaults) {
    if (raw.size() > 1 && raw[0] == '$') {
        const std::string key = raw.substr(1);
        const auto it = defaults.find(key);
        if (it != defaults.end())
            return it->second;
    }
    return raw;
}

static Vector3f parseVectorOrPoint(const pugi::xml_node& node,
                            const std::unordered_map<std::string, std::string>& defaults)
{
    std::string valStr = resolveValue(node.attribute("value").value(), defaults);
    if (!valStr.empty()) {
        return parseVec3(valStr);
    }
    float x = 0.f, y = 0.f, z = 0.f;
    std::string xStr = resolveValue(node.attribute("x").value(), defaults);
    std::string yStr = resolveValue(node.attribute("y").value(), defaults);
    std::string zStr = resolveValue(node.attribute("z").value(), defaults);
    if (!xStr.empty()) x = std::stof(xStr);
    if (!yStr.empty()) y = std::stof(yStr);
    if (!zStr.empty()) z = std::stof(zStr);
    return Vector3f(x, y, z);
}

static void parseProperty(const pugi::xml_node& node, PropertyList& list, const std::unordered_map<std::string, std::string>& defaults) {
    const std::string name = node.attribute("name").value();
    if (name.empty()) return;

    ETag tag = getTag(node.name());
    switch (tag) {
        case EBoolean: {
            std::string val = resolveValue(node.attribute("value").value(), defaults);
            list.setBoolean(name, val == "true" || val == "1");
            break;
        }
        case EInteger: {
            std::string val = resolveValue(node.attribute("value").value(), defaults);
            list.setInteger(name, std::stoi(val));
            break;
        }
        case EFloat: {
            std::string val = resolveValue(node.attribute("value").value(), defaults);
            list.setFloat(name, std::stof(val));
            break;
        }
        case EString: {
            std::string val = resolveValue(node.attribute("value").value(), defaults);
            list.setString(name, val);
            break;
        }
        case EColor: {
            Vector3f c = parseVectorOrPoint(node, defaults);
            list.setColor(name, Color3f(c.x, c.y, c.z));
            break;
        }
        case EPoint: {
            Vector3f p = parseVectorOrPoint(node, defaults);
            list.setPoint(name, Point3f(p.x, p.y, p.z));
            break;
        }
        case EVector: {
            Vector3f v = parseVectorOrPoint(node, defaults);
            list.setVector(name, Vector3f(v.x, v.y, v.z));
            break;
        }
        case ETexture: {
            std::string filename = resolveValue(node.attribute("filename").value(), defaults);
            if (filename.empty()) {
                for (pugi::xml_node ch : node.children()) {
                    if (getTag(ch.name()) == EString && std::string(ch.attribute("name").value()) == "filename") {
                        filename = resolveValue(ch.attribute("value").value(), defaults);
                    }
                }
            }
            if (!filename.empty()) {
                list.setString(name + "_texture", filename);
            }
            list.setColor(name, Color3f(1.f));
            break;
        }
        default:
            break;
    }
}

static void parseProperties(const pugi::xml_node& node, PropertyList& list, const std::unordered_map<std::string, std::string>& defaults) {
    for (pugi::xml_node child : node.children()) {
        parseProperty(child, list, defaults);
    }
}

static bool flattenBsdfNode(const pugi::xml_node& bsdfNode,
                     PropertyList& bsdfProps,
                     std::vector<std::string>& warnings,
                     const std::unordered_map<std::string, std::string>& defaults)
{
    if (!bsdfNode || getTag(bsdfNode.name()) != EBsdf)
        return false;

    const std::string type = bsdfNode.attribute("type").value();
    if (type == "twosided" || type == "bumpmap" || type == "mask" || type == "normalmap") {
        const pugi::xml_node inner = bsdfNode.child("bsdf");
        if (!inner) {
            warnings.push_back("Found <bsdf type='" + type + "'> without nested <bsdf>; using diffuse fallback.");
            bsdfProps.setString("type", "diffuse");
            return true;
        }
        return flattenBsdfNode(inner, bsdfProps, warnings, defaults);
    }

    bsdfProps.setString("type", type.empty() ? "diffuse" : type);
    parseProperties(bsdfNode, bsdfProps, defaults);
    return true;
}

static void parseTransform(const pugi::xml_node& node, Matrix4f& transform, Matrix4f& normalTransform, const std::unordered_map<std::string, std::string>& defaults) {
    for (const pugi::xml_node& op : node.children()) {
        ETag tag = getTag(op.name());
        switch (tag) {
            case ETranslate: {
                const Vector3f t = parseVectorOrPoint(op, defaults);
                transform = Matrix4f::translate(t) * transform;
                break;
            }
            case EScale: {
                const Vector3f s = parseVectorOrPoint(op, defaults);
                transform = Matrix4f::scale(s) * transform;
                const Vector3f invS(
                    std::abs(s.x) > 1e-9f ? 1.f / s.x : 1.f,
                    std::abs(s.y) > 1e-9f ? 1.f / s.y : 1.f,
                    std::abs(s.z) > 1e-9f ? 1.f / s.z : 1.f
                );
                normalTransform = Matrix4f::scale(invS) * normalTransform;
                break;
            }
            case ERotate: {
                Vector3f axis(0.f);
                std::string axisStr = resolveValue(op.attribute("axis").value(), defaults);
                if (!axisStr.empty()) {
                    axis = parseVec3(axisStr);
                } else {
                    std::string xStr = resolveValue(op.attribute("x").value(), defaults);
                    std::string yStr = resolveValue(op.attribute("y").value(), defaults);
                    std::string zStr = resolveValue(op.attribute("z").value(), defaults);
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
                const float angle = toF(resolveValue(op.attribute("angle").value(), defaults));
                transform   = Matrix4f::rotate(axis, angle) * transform;
                normalTransform = Matrix4f::rotate(axis, angle) * normalTransform;
                break;
            }
            case EMatrix: {
                Matrix4f explicitM;
                if (parseMatrix4fValue(resolveValue(op.attribute("value").value(), defaults), explicitM)) {
                    transform = explicitM;
                    normalTransform = inverseTransposeUpper3x3(explicitM);
                }
                break;
            }
            default:
                break;
        }
    }
}

static bool parseMesh(const std::string& baseDir,
               const std::string& objFilename,
               const std::string& meshName,
               int                materialId,
               int                emitterId,
               const Matrix4f&    transform,
               const Matrix4f&    normalTransform,
               CPUScene&          out,
               std::string&       errorOut)
{
    fs::path objPath = fs::path(baseDir) / objFilename;
    tinyobj::ObjReaderConfig reader_config;
    reader_config.triangulate = true;
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(objPath.string(), reader_config)) {
        if (!reader.Error().empty()) {
            errorOut = reader.Error();
        } else {
            errorOut = "Cannot open OBJ file: " + objPath.string();
        }
        return false;
    }

    if (!reader.Warning().empty()) {
        out.warnings.push_back("OBJ Warning (" + objFilename + "): " + reader.Warning());
    }

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();

    const uint32_t meshTriangleStart = (uint32_t)out.triangles.size();
    const int meshId = (int)out.meshes.size();
    
    AABB bbox;

    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
            if (fv != 3) {
                index_offset += fv;
                continue;
            }

            Triangle tri;
            tri.material_id = materialId;
            tri.mesh_id = meshId;

            Point3f pts[3];
            Vector3f norms[3];
            Point2f uvs[3];
            bool has_n = true;
            bool has_uv = true;

            for (size_t v = 0; v < 3; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                float vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                float vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                float vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                pts[v] = transform * Point3f(vx, vy, vz);

                bbox.grow(pts[v]);

                if (idx.normal_index >= 0) {
                    float nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    float ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    float nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                    norms[v] = normalize(normalTransform * Vector3f(nx, ny, nz));
                } else {
                    has_n = false;
                }

                if (idx.texcoord_index >= 0) {
                    float tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    float ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                    uvs[v] = Point2f(tx, ty);
                } else {
                    has_uv = false;
                }
            }

            tri.p0 = pts[0]; tri.p1 = pts[1]; tri.p2 = pts[2];
            if (has_n) {
                tri.n0 = norms[0]; tri.n1 = norms[1]; tri.n2 = norms[2];
                tri.has_normals = true;
            } else {
                tri.has_normals = false;
            }
            if (has_uv) {
                tri.uv0 = uvs[0]; tri.uv1 = uvs[1]; tri.uv2 = uvs[2];
                tri.has_uvs = true;
            } else {
                tri.has_uvs = false;
            }

            out.triangles.push_back(tri);
            index_offset += fv;
        }
    }

    if (out.triangles.size() == meshTriangleStart) {
        errorOut = "OBJ file has no triangles: " + objPath.string();
        return false;
    }

    const uint32_t meshTriangleCount = (uint32_t)out.triangles.size() - meshTriangleStart;

    MeshInstance meshInst;
    meshInst.materialId    = materialId;
    meshInst.triangleStart = meshTriangleStart;
    meshInst.triangleCount = meshTriangleCount;
    meshInst.transform     = transform;
    meshInst.emitterType   = (emitterId >= 0) ? EmitterType::Area : EmitterType::None;
    meshInst.emitterId     = emitterId;
    meshInst.bbox          = bbox;

    out.meshes.push_back(meshInst);
    out.meshNames.push_back(meshName);
    return true;
}

static bool parseMeshPLY(const std::string& baseDir,
                  const std::string& plyFilename,
                  const std::string& meshName,
                  int                materialId,
                  int                emitterId,
                  const Matrix4f&    transform,
                  const Matrix4f&    normalTransform,
                  CPUScene&          out,
                  std::string&       errorOut)
{
    fs::path plyPath = fs::path(baseDir) / plyFilename;
    try {
        happly::PLYData plyIn(plyPath.string());
        
        std::vector<std::array<double, 3>> vPos = plyIn.getVertexPositions();
        
        bool hasNormals = plyIn.getElement("vertex").hasProperty("nx") && 
                          plyIn.getElement("vertex").hasProperty("ny") && 
                          plyIn.getElement("vertex").hasProperty("nz");
        
        std::vector<std::array<double, 3>> vNormals;
        if (hasNormals) {
            vNormals.resize(vPos.size());
            std::vector<double> nx = plyIn.getElement("vertex").getProperty<double>("nx");
            std::vector<double> ny = plyIn.getElement("vertex").getProperty<double>("ny");
            std::vector<double> nz = plyIn.getElement("vertex").getProperty<double>("nz");
            for(size_t i=0; i<vPos.size(); i++) {
                vNormals[i] = {nx[i], ny[i], nz[i]};
            }
        }
        
        bool hasUVs = false;
        std::vector<std::array<double, 2>> vUVs;
        if (plyIn.getElement("vertex").hasProperty("u") && plyIn.getElement("vertex").hasProperty("v")) {
            hasUVs = true;
            std::vector<double> u = plyIn.getElement("vertex").getProperty<double>("u");
            std::vector<double> v = plyIn.getElement("vertex").getProperty<double>("v");
            vUVs.resize(vPos.size());
            for(size_t i=0; i<vPos.size(); i++) vUVs[i] = {u[i], v[i]};
        } else if (plyIn.getElement("vertex").hasProperty("s") && plyIn.getElement("vertex").hasProperty("t")) {
            hasUVs = true;
            std::vector<double> s = plyIn.getElement("vertex").getProperty<double>("s");
            std::vector<double> t = plyIn.getElement("vertex").getProperty<double>("t");
            vUVs.resize(vPos.size());
            for(size_t i=0; i<vPos.size(); i++) vUVs[i] = {s[i], t[i]};
        }

        std::vector<std::vector<size_t>> fInd = plyIn.getFaceIndices<size_t>();

        const uint32_t meshTriangleStart = (uint32_t)out.triangles.size();
        const int meshId = (int)out.meshes.size();
        
        AABB bbox;

        for (const auto& face : fInd) {
            if (face.size() < 3) continue;
            for (size_t i = 1; i + 1 < face.size(); ++i) {
                Triangle tri;
                tri.material_id = materialId;
                tri.mesh_id = meshId;

                size_t idx0 = face[0];
                size_t idx1 = face[i];
                size_t idx2 = face[i+1];

                tri.p0 = transform * Point3f((float)vPos[idx0][0], (float)vPos[idx0][1], (float)vPos[idx0][2]);
                tri.p1 = transform * Point3f((float)vPos[idx1][0], (float)vPos[idx1][1], (float)vPos[idx1][2]);
                tri.p2 = transform * Point3f((float)vPos[idx2][0], (float)vPos[idx2][1], (float)vPos[idx2][2]);

                bbox.grow(tri.p0);
                bbox.grow(tri.p1);
                bbox.grow(tri.p2);

                if (hasNormals) {
                    tri.n0 = normalize(normalTransform * Vector3f((float)vNormals[idx0][0], (float)vNormals[idx0][1], (float)vNormals[idx0][2]));
                    tri.n1 = normalize(normalTransform * Vector3f((float)vNormals[idx1][0], (float)vNormals[idx1][1], (float)vNormals[idx1][2]));
                    tri.n2 = normalize(normalTransform * Vector3f((float)vNormals[idx2][0], (float)vNormals[idx2][1], (float)vNormals[idx2][2]));
                    tri.has_normals = true;
                } else {
                    tri.has_normals = false;
                }

                if (hasUVs) {
                    tri.uv0 = Point2f((float)vUVs[idx0][0], (float)vUVs[idx0][1]);
                    tri.uv1 = Point2f((float)vUVs[idx1][0], (float)vUVs[idx1][1]);
                    tri.uv2 = Point2f((float)vUVs[idx2][0], (float)vUVs[idx2][1]);
                    tri.has_uvs = true;
                } else {
                    tri.has_uvs = false;
                }

                out.triangles.push_back(tri);
            }
        }

        if (out.triangles.size() == meshTriangleStart) {
            errorOut = "PLY file has no triangles: " + plyPath.string();
            return false;
        }

        const uint32_t meshTriangleCount = (uint32_t)out.triangles.size() - meshTriangleStart;

        MeshInstance meshInst;
        meshInst.materialId    = materialId;
        meshInst.triangleStart = meshTriangleStart;
        meshInst.triangleCount = meshTriangleCount;
        meshInst.transform     = transform;
        meshInst.emitterType   = (emitterId >= 0) ? EmitterType::Area : EmitterType::None;
        meshInst.emitterId     = emitterId;
        meshInst.bbox          = bbox;

        out.meshes.push_back(meshInst);
        out.meshNames.push_back(meshName);
        return true;
    } catch (const std::exception& e) {
        errorOut = std::string("Error parsing PLY: ") + e.what();
        return false;
    }
}

static bool parseCamera(const std::string& originStr,
                 const std::string& targetStr,
                 const std::string& upStr,
                 float              fov,
                 CPUScene&          out,
                 std::string&       /*errorOut*/)
{
    const Vector3f o = parseVec3(originStr);
    const Vector3f t = parseVec3(targetStr);
    const Vector3f u = parseVec3(upStr);

    out.camera.origin = Point3f(o.x, o.y, o.z);
    out.camera.target = Point3f(t.x, t.y, t.z);
    out.camera.up     = u;
    out.camera.fov    = fov;
    out.camera.hasCamera = true;
    return true;
}

} // anonymous namespace

bool SceneLoader::load(const std::string& xmlPath,
                       CPUScene&          out,
                       std::string&       errorOut)
{
    out = CPUScene();

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

    // First pass: collect top-level BSDF definitions
    for (const pugi::xml_node& node : root.children()) {
        if (getTag(node.name()) != EBsdf)
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
        if (!flattenBsdfNode(node, bsdfProps, out.warnings, defaults))
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
        ETag tag = getTag(node.name());

        switch (tag) {
            case EMesh: {
                try {
                    PropertyList meshProps;
                    parseProperties(node, meshProps, defaults);

                    const std::string shapeType = (std::string(node.name()) == "shape")
                        ? std::string(node.attribute("type").value())
                        : std::string("obj");

                    std::string objFile;
                    if (shapeType == "obj" || shapeType == "ply" || std::string(node.name()) == "mesh")
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

                    Matrix4f meshTransform;
                    Matrix4f normalTransform;

                    for (const pugi::xml_node& child : node.children()) {
                        ETag cTag = getTag(child.name());
                        switch (cTag) {
                            case EBsdf:
                                flattenBsdfNode(child, bsdfProps, out.warnings, defaults);
                                break;
                            case ERef: {
                                const std::string refId = child.attribute("id").value();
                                auto it = bsdfIdToMaterial.find(refId);
                                if (it != bsdfIdToMaterial.end()) {
                                    materialId = it->second;
                                } else {
                                    out.warnings.push_back("Shape references unknown BSDF id '" + refId + "'; using diffuse fallback.");
                                }
                                break;
                            }
                            case EEmitter: {
                                const std::string emitterType = child.attribute("type").value();
                                parseProperties(child, emitterProps, defaults);

                                EmitterInstance inst = makeEmitterFromPropertyLists(
                                    emitterType, emitterProps, out.warnings);
                                if (inst.type != EmitterType::None) {
                                    emitterId = (int)out.emitters.size();
                                    out.emitters.push_back(inst);
                                }
                                break;
                            }
                            case EBoolean: {
                                const std::string propName = child.attribute("name").value();
                                if (propName == "face_normals") {
                                    const bool faceNormals = std::string(child.attribute("value").value()) == "true"
                                                          || std::string(child.attribute("value").value()) == "1";
                                    if (faceNormals) {
                                        out.warnings.push_back("Per-shape 'face_normals' requested; renderer currently uses global normal interpolation toggle.");
                                    }
                                }
                                break;
                            }
                            case ETransform: {
                                parseTransform(child, meshTransform, normalTransform, defaults);
                                break;
                            }
                            default:
                                break;
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

                    if (shapeType == "obj" || std::string(node.name()) == "mesh") {
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



                    ++nextMatId;

                } catch (const std::exception& e) {
                    errorOut = std::string("Mesh parse error: ") + e.what();
                    return false;
                }
                break;
            }
            case EEmitter: {
                const std::string emitterType = node.attribute("type").value();
                if (emitterType == "envmap") {
                    PropertyList envProps;
                    parseProperties(node, envProps, defaults);

                    Matrix4f envTransform;
                    bool hasTransform = false;
                    for (const pugi::xml_node& child : node.children()) {
                        if (getTag(child.name()) != ETransform)
                            continue;
                        const pugi::xml_node mnode = child.child("matrix");
                        if (mnode && parseMatrix4fValue(resolveValue(mnode.attribute("value").value(), defaults), envTransform)) {
                            hasTransform = true;
                        }
                    }

                    const std::string filename = envProps.getString("filename", std::string());
                    if (filename.empty()) {
                        out.warnings.push_back("envmap emitter missing filename; background will remain black.");
                        continue;
                    }

                    if (!loadEnvMapHDR(filename, baseDir, out.envMap.pixels, out.envMap.width, out.envMap.height, errorOut))
                        return false;

                    out.envMap.hasEnvMap = true;
                    out.envMap.toWorld = hasTransform ? envTransform : Matrix4f();
                } else if (emitterType == "constant") {
                    PropertyList envProps;
                    parseProperties(node, envProps, defaults);
                    const Color3f radiance = envProps.getColor("radiance",
                                               envProps.getColor("emission", Color3f(0.f)));
                    out.envMap.hasConstant = true;
                    out.envMap.constantRadiance = radiance;
                }
                break;
            }
            case ECamera: {
                PropertyList cameraProps;
                parseProperties(node, cameraProps, defaults);

                float fov = cameraProps.getFloat("fov", 45.f);
                std::string originStr, targetStr, upStr;
                Matrix4f sensorToWorld;
                bool hasSensorMatrix = false;

                for (const pugi::xml_node& child : node.children()) {
                    ETag cTag = getTag(child.name());
                    if (cTag == EFloat) {
                        if (std::string(child.attribute("name").value()) == "fov")
                            fov = toF(resolveValue(child.attribute("value").value(), defaults));
                    } else if (cTag == ETransform) {
                        const pugi::xml_node lookat = child.child("lookat");
                        if (lookat) {
                            originStr = resolveValue(lookat.attribute("origin").value(), defaults);
                            targetStr = resolveValue(lookat.attribute("target").value(), defaults);
                            upStr     = resolveValue(lookat.attribute("up").value(), defaults);
                        } else {
                            const pugi::xml_node mnode = child.child("matrix");
                            if (mnode && parseMatrix4fValue(resolveValue(mnode.attribute("value").value(), defaults), sensorToWorld)) {
                                hasSensorMatrix = true;
                            }
                        }
                    }
                }

                if (!originStr.empty()) {
                    if (!parseCamera(originStr, targetStr, upStr, fov, out, errorOut))
                        return false;
                } else if (hasSensorMatrix) {
                    const Point3f camO   = sensorToWorld * Point3f(0.f, 0.f, 0.f);
                    const Point3f camFwdP = sensorToWorld * Point3f(0.f, 0.f, 1.f);
                    const Vector3f camUpV = sensorToWorld * Vector3f(0.f, 1.f, 0.f);
                    const Vector3f camDir = normalize(camFwdP - camO);
                    const Vector3f camUp   = normalize(camUpV);

                    out.camera.origin = camO;
                    out.camera.target = camO + camDir;
                    out.camera.up     = camUp;
                    out.camera.fov    = fov;
                    out.camera.hasCamera = true;
                }
                break;
            }
            case EIntegrator:
                out.integratorType = resolveValue(node.attribute("type").value(), defaults);
                break;

            default:
                break;
        }
    }

    if (out.triangles.empty()) {
        errorOut = "Scene contains no geometry.";
        return false;
    }
    return true;
}

FUTABA_NAMESPACE_END