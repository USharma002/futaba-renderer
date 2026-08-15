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
    EMedium,

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
    ELookat,

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
        { "medium",     EMedium },
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
        { "matrix",     EMatrix },
        { "lookat",     ELookat }
    };
    auto it = tags.find(name);
    return (it != tags.end()) ? it->second : EInvalid;
}

static inline const char* skipSeparators(const char* p) {
    while (*p && (*p == ' ' || *p == ',' || *p == '\t' || *p == '\r' || *p == '\n')) {
        ++p;
    }
    return p;
}

static inline bool parseNextFloat(const char*& p, float& out) {
    p = skipSeparators(p);
    if (!*p) return false;
    char* end = nullptr;
    out = std::strtof(p, &end);
    if (p == end) return false;
    p = end;
    return true;
}

static inline Vector3f parseVec3(const std::string& s) {
    const char* p = s.c_str();
    float v[3] = {0.f, 0.f, 0.f};
    int count = 0;
    while (count < 3 && parseNextFloat(p, v[count])) {
        ++count;
    }
    if (count == 0) return Vector3f(0.f);
    if (count == 1) return Vector3f(v[0]);
    if (count == 2) return Vector3f(v[0], v[1], 0.f);
    return Vector3f(v[0], v[1], v[2]);
}

static inline bool parseMatrix4fValue(const std::string& s, Matrix4f& outM) {
    const char* p = s.c_str();
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            if (!parseNextFloat(p, outM.m[r][c]))
                return false;
        }
    }
    return true;
}

static inline float toF(const std::string& s) {
    const char* p = s.c_str();
    float v = 0.f;
    parseNextFloat(p, v);
    return v;
}

static inline std::string resolveValue(const std::string& raw, const std::unordered_map<std::string, std::string>& defaults) {
    if (raw.size() > 1 && raw[0] == '$') {
        const auto it = defaults.find(raw.substr(1));
        if (it != defaults.end())
            return it->second;
    }
    return raw;
}

static inline Vector3f parseVectorOrPoint(const pugi::xml_node& node,
                                         const std::unordered_map<std::string, std::string>& defaults)
{
    const char* valAttr = node.attribute("value").value();
    if (valAttr && *valAttr) {
        return parseVec3(resolveValue(valAttr, defaults));
    }
    float x = 0.f, y = 0.f, z = 0.f;
    const char* xAttr = node.attribute("x").value();
    const char* yAttr = node.attribute("y").value();
    const char* zAttr = node.attribute("z").value();
    if (xAttr && *xAttr) x = std::strtof(resolveValue(xAttr, defaults).c_str(), nullptr);
    if (yAttr && *yAttr) y = std::strtof(resolveValue(yAttr, defaults).c_str(), nullptr);
    if (zAttr && *zAttr) z = std::strtof(resolveValue(zAttr, defaults).c_str(), nullptr);
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
            list.setInteger(name, std::strtol(val.c_str(), nullptr, 10));
            break;
        }
        case EFloat: {
            std::string val = resolveValue(node.attribute("value").value(), defaults);
            list.setFloat(name, toF(val));
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

static void extractMaterialTextures(const PropertyList& bsdfProps,
                                   std::string& texPath,
                                   std::string& nmPath)
{
    texPath.clear();
    nmPath.clear();

    if (bsdfProps.hasProperty("reflectance_texture")) texPath = bsdfProps.getString("reflectance_texture");
    else if (bsdfProps.hasProperty("diffuse_reflectance_texture")) texPath = bsdfProps.getString("diffuse_reflectance_texture");
    else if (bsdfProps.hasProperty("specular_reflectance_texture")) texPath = bsdfProps.getString("specular_reflectance_texture");

    if (texPath.empty()) {
        for (const auto& [propName, propVal] : bsdfProps.getProperties()) {
            if (const auto* s = std::get_if<std::string>(&propVal)) {
                std::string valLower = *s;
                for (auto& c : valLower) c = std::tolower(c);
                if (propName.find("texture") != std::string::npos ||
                    valLower.find(".png") != std::string::npos ||
                    valLower.find(".jpg") != std::string::npos ||
                    valLower.find(".tga") != std::string::npos ||
                    valLower.find(".bmp") != std::string::npos) {
                    texPath = *s;
                    break;
                }
            }
        }
    }

    if (bsdfProps.hasProperty("normalmap_texture")) nmPath = bsdfProps.getString("normalmap_texture");
    else if (bsdfProps.hasProperty("normal_texture")) nmPath = bsdfProps.getString("normal_texture");
    else nmPath = bsdfProps.getString("normalmap", bsdfProps.getString("normal", ""));

    if (nmPath.empty()) {
        for (const auto& [propName, propVal] : bsdfProps.getProperties()) {
            if (const auto* s = std::get_if<std::string>(&propVal)) {
                if (propName.find("normal") != std::string::npos && *s != texPath) {
                    nmPath = *s;
                    break;
                }
            }
        }
    }
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
                axis = (len > 1e-6f) ? (axis / len) : Vector3f(1.f, 0.f, 0.f);
                const float angle = toF(resolveValue(op.attribute("angle").value(), defaults));
                transform   = Matrix4f::rotate(axis, angle) * transform;
                normalTransform = Matrix4f::rotate(axis, angle) * normalTransform;
                break;
            }
            case EMatrix: {
                Matrix4f explicitM;
                if (parseMatrix4fValue(resolveValue(op.attribute("value").value(), defaults), explicitM)) {
                    transform = explicitM * transform;
                    normalTransform = Matrix4f::inverseTransposeUpper3x3(explicitM) * normalTransform;
                }
                break;
            }
            case ELookat: {
                std::string oStr = resolveValue(op.attribute("origin").value(), defaults);
                std::string tStr = resolveValue(op.attribute("target").value(), defaults);
                std::string uStr = resolveValue(op.attribute("up").value(), defaults);
                if (!oStr.empty() && !tStr.empty()) {
                    Vector3f o = parseVec3(oStr), t = parseVec3(tStr), u = !uStr.empty() ? parseVec3(uStr) : Vector3f(0.f, 1.f, 0.f);
                    Matrix4f lookatM = Matrix4f::lookAt(Point3f(o.x, o.y, o.z), Point3f(t.x, t.y, t.z), u);
                    transform = lookatM * transform;
                    normalTransform = Matrix4f::inverseTransposeUpper3x3(lookatM) * normalTransform;
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

    std::vector<Triangle> localTriangles;

    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
            if (fv != 3) {
                index_offset += fv;
                continue;
            }

            Triangle tri;
            Point3f pts[3];
            Vector3f norms[3];
            Point2f uvs[3];
            bool has_n = true, has_uv = true;

            for (size_t v = 0; v < 3; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                pts[v] = Point3f(attrib.vertices[3 * size_t(idx.vertex_index) + 0],
                                 attrib.vertices[3 * size_t(idx.vertex_index) + 1],
                                 attrib.vertices[3 * size_t(idx.vertex_index) + 2]);

                if (idx.normal_index >= 0) {
                    norms[v] = Vector3f(attrib.normals[3 * size_t(idx.normal_index) + 0],
                                        attrib.normals[3 * size_t(idx.normal_index) + 1],
                                        attrib.normals[3 * size_t(idx.normal_index) + 2]);
                } else {
                    has_n = false;
                }

                if (idx.texcoord_index >= 0) {
                    uvs[v] = Point2f(attrib.texcoords[2 * size_t(idx.texcoord_index) + 0],
                                     attrib.texcoords[2 * size_t(idx.texcoord_index) + 1]);
                } else {
                    has_uv = false;
                }
            }

            tri.p0 = pts[0]; tri.p1 = pts[1]; tri.p2 = pts[2];
            if (has_n) {
                tri.n0 = norms[0]; tri.n1 = norms[1]; tri.n2 = norms[2];
                tri.has_normals = true;
            }
            if (has_uv) {
                tri.uv0 = uvs[0]; tri.uv1 = uvs[1]; tri.uv2 = uvs[2];
                tri.has_uvs = true;
            }

            localTriangles.push_back(tri);
            index_offset += fv;
        }
    }

    if (localTriangles.empty()) {
        errorOut = "OBJ file has no triangles: " + objPath.string();
        return false;
    }

    return appendMeshGeometry(meshName, materialId, emitterId, transform, normalTransform, localTriangles, out);
}

static void computeMeshVertexNormals(const std::vector<std::array<double, 3>>& vPos,
                                     const std::vector<std::vector<size_t>>& fInd,
                                     std::vector<Vector3f>& computedNormals)
{
    computedNormals.assign(vPos.size(), Vector3f(0.f));
    for (const auto& face : fInd) {
        if (face.size() < 3) continue;
        for (size_t i = 1; i + 1 < face.size(); ++i) {
            size_t idx0 = face[0];
            size_t idx1 = face[i];
            size_t idx2 = face[i+1];
            Point3f p0((float)vPos[idx0][0], (float)vPos[idx0][1], (float)vPos[idx0][2]);
            Point3f p1((float)vPos[idx1][0], (float)vPos[idx1][1], (float)vPos[idx1][2]);
            Point3f p2((float)vPos[idx2][0], (float)vPos[idx2][1], (float)vPos[idx2][2]);
            Vector3f fn = cross(p1 - p0, p2 - p0);
            computedNormals[idx0] += fn;
            computedNormals[idx1] += fn;
            computedNormals[idx2] += fn;
        }
    }
    for (auto& n : computedNormals) {
        if (n.lengthSquared() > 0.f)
            n = normalize(n);
    }
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
        std::vector<Triangle> localTriangles;

        std::vector<Vector3f> computedNormals;
        if (!hasNormals) {
            computeMeshVertexNormals(vPos, fInd, computedNormals);
        }

        for (const auto& face : fInd) {
            if (face.size() < 3) continue;
            for (size_t i = 1; i + 1 < face.size(); ++i) {
                Triangle tri;
                size_t idx0 = face[0];
                size_t idx1 = face[i];
                size_t idx2 = face[i+1];

                tri.p0 = Point3f((float)vPos[idx0][0], (float)vPos[idx0][1], (float)vPos[idx0][2]);
                tri.p1 = Point3f((float)vPos[idx1][0], (float)vPos[idx1][1], (float)vPos[idx1][2]);
                tri.p2 = Point3f((float)vPos[idx2][0], (float)vPos[idx2][1], (float)vPos[idx2][2]);

                if (hasNormals) {
                    tri.n0 = Vector3f((float)vNormals[idx0][0], (float)vNormals[idx0][1], (float)vNormals[idx0][2]);
                    tri.n1 = Vector3f((float)vNormals[idx1][0], (float)vNormals[idx1][1], (float)vNormals[idx1][2]);
                    tri.n2 = Vector3f((float)vNormals[idx2][0], (float)vNormals[idx2][1], (float)vNormals[idx2][2]);
                } else {
                    tri.n0 = computedNormals[idx0];
                    tri.n1 = computedNormals[idx1];
                    tri.n2 = computedNormals[idx2];
                }
                tri.has_normals = true;

                if (hasUVs) {
                    tri.uv0 = Point2f((float)vUVs[idx0][0], (float)vUVs[idx0][1]);
                    tri.uv1 = Point2f((float)vUVs[idx1][0], (float)vUVs[idx1][1]);
                    tri.uv2 = Point2f((float)vUVs[idx2][0], (float)vUVs[idx2][1]);
                    tri.has_uvs = true;
                }

                localTriangles.push_back(tri);
            }
        }

        if (localTriangles.empty()) {
            errorOut = "PLY file has no triangles: " + plyPath.string();
            return false;
        }

        return appendMeshGeometry(meshName, materialId, emitterId, transform, normalTransform, localTriangles, out);
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

static Medium parseMediumNode(const pugi::xml_node& node, const std::unordered_map<std::string, std::string>& defaults) {
    PropertyList props;
    parseProperties(node, props, defaults);

    float scale = props.getFloat("scale", 1.0f);
    float g = props.getFloat("g", 0.0f);

    for (const pugi::xml_node& child : node.children()) {
        if (std::string(child.name()) == "phase") {
            PropertyList phaseProps;
            parseProperties(child, phaseProps, defaults);
            g = phaseProps.getFloat("g", g);
        }
    }

    Color3f sigma_s(0.f), sigma_a(0.f), sigma_t(0.f);

    if (props.hasProperty("sigma_t") || props.hasProperty("albedo")) {
        sigma_t = props.getColor("sigma_t", Color3f(1.f)) * scale;
        Color3f alb = props.getColor("albedo", Color3f(1.f));
        sigma_s = sigma_t * alb;
        sigma_a = Color3f(fmaxf(0.f, sigma_t.x - sigma_s.x),
                          fmaxf(0.f, sigma_t.y - sigma_s.y),
                          fmaxf(0.f, sigma_t.z - sigma_s.z));
    } else {
        sigma_s = props.getColor("sigma_s", Color3f(0.f)) * scale;
        sigma_a = props.getColor("sigma_a", Color3f(0.f)) * scale;
        sigma_t = sigma_s + sigma_a;
    }

    return Medium(sigma_a, sigma_s, g);
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
    std::unordered_map<std::string, int> mediumIdToMedium;
    std::unordered_map<std::string, std::string> defaults;

    for (const pugi::xml_node& node : root.children()) {
        if (std::string(node.name()) != "default")
            continue;
        const std::string key = node.attribute("name").value();
        const std::string val = node.attribute("value").value();
        if (!key.empty())
            defaults[key] = val;
    }

    // Collect top-level Medium definitions
    for (const pugi::xml_node& node : root.children()) {
        if (getTag(node.name()) != EMedium)
            continue;

        std::string medId = node.attribute("id").value();
        if (medId.empty())
            continue;

        Medium med = parseMediumNode(node, defaults);
        int medIdx = (int)out.media.size();
        out.media.push_back(med);
        out.mediumNames.push_back(medId);
        mediumIdToMedium[medId] = medIdx;
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

        std::string texPath, nmPath;
        extractMaterialTextures(bsdfProps, texPath, nmPath);
        out.materialTexturePaths.push_back(texPath);
        out.materialNormalMapPaths.push_back(nmPath);
    }

    for (const pugi::xml_node& node : root.children()) {
        ETag tag = getTag(node.name());

        switch (tag) {
            case EMesh: {
                try {
                    PropertyList meshProps;
                    parseProperties(node, meshProps, defaults);

                    std::string shapeType = std::string(node.attribute("type").value());
                    std::string objFile = meshProps.getString("filename");

                    if (shapeType.empty() || shapeType == "mesh") {
                        if (fs::path(objFile).extension() == ".ply") {
                            shapeType = "ply";
                        } else {
                            shapeType = "obj";
                        }
                    }

                    std::string meshName = node.attribute("id").value();
                    if (meshName.empty()) {
                        meshName = (objFile.empty() ? shapeType : fs::path(objFile).stem().string());
                        if (meshName.empty())
                            meshName = "mesh_" + std::to_string(nextMatId);
                    }

                    PropertyList bsdfProps, emitterProps;
                    int materialId = -1;
                    int emitterId = -1;
                    int interiorMediumId = -1;
                    int exteriorMediumId = -1;

                    Matrix4f meshTransform;
                    Matrix4f normalTransform;

                    for (const pugi::xml_node& child : node.children()) {
                        ETag cTag = getTag(child.name());
                        switch (cTag) {
                            case EBsdf:
                                flattenBsdfNode(child, bsdfProps, out.warnings, defaults);
                                break;
                            case EMedium: {
                                std::string medName = child.attribute("name").value();
                                Medium med = parseMediumNode(child, defaults);
                                int medIdx = (int)out.media.size();
                                out.media.push_back(med);
                                out.mediumNames.push_back(medName.empty() ? ("med_" + std::to_string(medIdx)) : medName);
                                if (medName == "exterior") exteriorMediumId = medIdx;
                                else interiorMediumId = medIdx;
                                break;
                            }
                            case ERef: {
                                const std::string refId = child.attribute("id").value();
                                const std::string refName = child.attribute("name").value();
                                auto itM = mediumIdToMedium.find(refId);
                                if (itM != mediumIdToMedium.end()) {
                                    if (refName == "exterior") exteriorMediumId = itM->second;
                                    else interiorMediumId = itM->second;
                                } else {
                                    auto it = bsdfIdToMaterial.find(refId);
                                    if (it != bsdfIdToMaterial.end()) {
                                        materialId = it->second;
                                    } else {
                                        out.warnings.push_back("Shape references unknown BSDF/Medium id '" + refId + "'; using diffuse fallback.");
                                    }
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

                        std::string texPath, nmPath;
                        extractMaterialTextures(bsdfProps, texPath, nmPath);
                        out.materialTexturePaths.push_back(texPath);
                        out.materialNormalMapPaths.push_back(nmPath);
                    }

                    if (materialId >= 0 && (size_t)materialId < out.materials.size()) {
                        if (interiorMediumId >= 0) out.materials[materialId].interiorMediumId = interiorMediumId;
                        if (exteriorMediumId >= 0) out.materials[materialId].exteriorMediumId = exteriorMediumId;
                    }

                    if (shapeType == "ply" || fs::path(objFile).extension() == ".ply") {
                        if (!parseMeshPLY(baseDir, objFile, meshName, materialId,
                                          emitterId, meshTransform, normalTransform, out, errorOut))
                            return false;
                    } else if (shapeType == "obj" || shapeType == "mesh" || std::string(node.name()) == "mesh") {
                        if (!parseMesh(baseDir, objFile, meshName, materialId,
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