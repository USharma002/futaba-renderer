#include "scene_uploader.h"
#include "distribution.cuh"
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

namespace futaba {

UploadedSceneConfig SceneUploader::upload(
    const LoadedScene& loaded,
    const std::string& xmlPath,
    Scene& scene,
    TextureManager& textureManager,
    bool useVertexNormals,
    bool useNEE)
{
    std::string baseDir = fs::path(xmlPath).parent_path().string();
    if (baseDir.empty()) baseDir = ".";

    // We make a mutable copy of LoadedScene's materials to bind loaded texture objects
    std::vector<Material> tempMaterials = loaded.materials;
    for (size_t i = 0; i < tempMaterials.size(); ++i) {
        if (i < loaded.materialTexturePaths.size() && !loaded.materialTexturePaths[i].empty()) {
            fs::path texPath = fs::path(baseDir) / loaded.materialTexturePaths[i];
            cudaTextureObject_t texObj = textureManager.createTexture(texPath.string());
            tempMaterials[i].texObj = texObj;
        }
    }

    scene.setTriangles(loaded.triangles.data(), (uint32_t)loaded.triangles.size());
    scene.setMaterials(tempMaterials.data(), (uint32_t)tempMaterials.size());
    
    std::vector<futaba::EmitterGPU> emittersGPU;
    emittersGPU.reserve(loaded.emitters.size());
    for (const auto& emitter : loaded.emitters) {
        futaba::EmitterGPU g;
        g.type = static_cast<uint32_t>(emitter.type);
        g.flags = emitter.twoSided ? futaba::EMITTER_FLAG_TWO_SIDED : 0u;
        g.radiance = emitter.radiance;
        g.position = emitter.position;
        g.direction = emitter.direction;
        g.attachedMeshId = -1;
        emittersGPU.push_back(g);
    }

    // Convert and upload mesh instances
    std::vector<futaba::MeshInstanceGPU> meshGPU;
    for (size_t i = 0; i < loaded.meshes.size(); ++i) {
        const auto& mesh = loaded.meshes[i];
        futaba::MeshInstanceGPU m;
        m.triangleStart = mesh.triangleStart;
        m.triangleCount = mesh.triangleCount;
        m.emitterId = mesh.emitterId;
        meshGPU.push_back(m);

        if (mesh.emitterId >= 0 && mesh.emitterId < (int)emittersGPU.size()) {
            emittersGPU[mesh.emitterId].attachedMeshId = (int)i;
        }
    }
    scene.setMeshes(meshGPU.data(), (uint32_t)meshGPU.size());
    scene.setEmitters(emittersGPU.data(), (uint32_t)emittersGPU.size());

    // Build emissive-triangle distribution (area * emission luminance)
    std::vector<int> emissiveTriIndices;
    std::vector<float> emissiveWeights;
    emissiveTriIndices.reserve(loaded.triangles.size());
    emissiveWeights.reserve(loaded.triangles.size());

    auto luminance = [](const Color3f &c) {
        return futaba::getLuminance(c);
    };

    auto triangleEmission = [&](const Triangle &t) -> Color3f {
        if (t.mesh_id >= 0 && t.mesh_id < (int)loaded.meshes.size()) {
            const int emitterId = loaded.meshes[t.mesh_id].emitterId;
            if (emitterId >= 0 && emitterId < (int)loaded.emitters.size()) {
                return loaded.emitters[emitterId].radiance;
            }
        }

        if (t.material_id >= 0 && t.material_id < (int)loaded.materials.size()) {
            return loaded.materials[t.material_id].emission;
        }

        return Color3f(0.f);
    };

    for (size_t i = 0; i < loaded.triangles.size(); ++i) {
        const Triangle &t = loaded.triangles[i];
        float area = t.area();
        Color3f emission = triangleEmission(t);
        float w = area * luminance(emission);
        if (w > 0.f) {
            emissiveTriIndices.push_back((int)i);
            emissiveWeights.push_back(w);
        }
    }

    if (!emissiveWeights.empty()) {
        futaba::Distribution1D dist;
        dist.build(emissiveWeights);
        // Build global->emissive index map
        std::vector<int> globalToEmissive(loaded.triangles.size(), -1);
        for (size_t i = 0; i < emissiveTriIndices.size(); ++i) {
            int g = emissiveTriIndices[i];
            if (g >= 0 && g < (int)globalToEmissive.size())
                globalToEmissive[g] = (int)i;
        }
        // Upload cdf and index list to device (also provide global mapping)
        scene.setEmitterTriangleDistribution(dist.cdfData(), (int)dist.cdf.size(), dist.funcSum, emissiveTriIndices.data(), (int)emissiveTriIndices.size(), globalToEmissive.data(), (int)globalToEmissive.size());
    } else {
        scene.setEmitterTriangleDistribution(nullptr, 0, 0.f, nullptr, 0, nullptr, 0);
    }

    // Upload non-area (point / directional) emitter indices for NEE sampling
    {
        std::vector<int> nonAreaIndices;
        for (size_t i = 0; i < emittersGPU.size(); ++i) {
            const uint32_t t = emittersGPU[i].type;
            if (t == futaba::kEmitterTypePoint || t == futaba::kEmitterTypeDirectional)
                nonAreaIndices.push_back((int)i);
        }
        if (!nonAreaIndices.empty())
            scene.setNonAreaEmitters(nonAreaIndices.data(), (int)nonAreaIndices.size());
        else
            scene.setNonAreaEmitters(nullptr, 0);
    }

    if (loaded.hasEnvMap) {
        scene.setEnvironmentMap(loaded.envMapPixels.data(),
                                (uint32_t)loaded.envMapWidth,
                                (uint32_t)loaded.envMapHeight,
                                loaded.envMapToWorld);
    } else if (loaded.hasConstantEnv) {
        scene.setConstantEnvironment(loaded.constantEnv);
    } else {
        scene.setEnvironmentMap(nullptr, 0, 0, ::Matrix4f());
    }
    
    scene.use_vertex_normals = useVertexNormals;
    scene.use_nee = useNEE;

    UploadedSceneConfig config;
    config.hasMedium = loaded.hasMedium;
    config.mediumMeshId = loaded.mediumMeshId;
    config.medium = Medium(MEDIUM_HOMOGENEOUS, HomogeneousMedium(loaded.mediumSigmaS, loaded.mediumSigmaA, loaded.mediumG));
    config.integratorType = loaded.integratorType;
    config.triangleCount = (uint32_t)loaded.triangles.size();

    if (loaded.hasCamera) {
        config.hasCamera = true;
        config.camOrigin = loaded.camOrigin;
        config.camTarget = loaded.camTarget;
        config.camUp = loaded.camUp;
        config.camFov = loaded.camFov;
        
        ::Vector3f toTarget(loaded.camTarget.x - loaded.camOrigin.x,
                            loaded.camTarget.y - loaded.camOrigin.y,
                            loaded.camTarget.z - loaded.camOrigin.z);
        float loadedFocusDistance = toTarget.length();
        if (loadedFocusDistance > 0.f)
            config.currentFocusDistance = loadedFocusDistance;
    }

    return config;
}

} // namespace futaba
