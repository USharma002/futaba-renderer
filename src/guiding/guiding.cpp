#include "guiding.h"
#include <iostream>

namespace futaba {

GuidingManager::GuidingManager()
    : m_mode(PATH_GUIDING_NONE), m_width(0), m_height(0), m_sTree(nullptr), m_iteration(0) {}

GuidingManager::~GuidingManager() {
    destroy();
}

bool GuidingManager::init(int width, int height) {
    m_width = width;
    m_height = height;
    
    AABB bounds;
    bounds.minP = Point3f(-10.f, -10.f, -10.f); 
    bounds.maxP = Point3f(10.f, 10.f, 10.f);
    
    m_sTree = new STree(bounds); 
    return true;
}

void GuidingManager::resize(int width, int height) {
    m_width = width;
    m_height = height;
}

void GuidingManager::reset() {
    if (m_sTree) {
        m_sTree->clear();
        m_sTree->initialize();
    }
    m_iteration = 0;
}

void GuidingManager::train(const TrainingBufferManager& bufferManager, int maxDepth) {
    if (m_mode != PATH_GUIDING_PPG || !m_sTree) return;

    size_t totalElements = static_cast<size_t>(m_width) * m_height * maxDepth;
    
    std::vector<float> hostActive(totalElements);
    std::vector<Point3f> hostPosition(totalElements);
    std::vector<Color3f> hostRadiance(totalElements);
    std::vector<Color3f> hostWo(totalElements);

    cudaMemcpy(hostActive.data(), bufferManager.getActive(), totalElements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostPosition.data(), bufferManager.getPosition(), totalElements * sizeof(Point3f), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostRadiance.data(), bufferManager.getRadiance(), totalElements * sizeof(Color3f), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostWo.data(), bufferManager.getWo(), totalElements * sizeof(Color3f), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < totalElements; ++i) {
        if (hostActive[i] > 0.5f) {
            DTreeRecord rec;
            rec.d = hostWo[i];
            rec.radiance = 0.2126f * hostRadiance[i].x + 0.7152f * hostRadiance[i].y + 0.0722f * hostRadiance[i].z;
            rec.woPdf = 1.0f; 
            rec.statisticalWeight = 1.0f;
            rec.isDelta = false;

            m_sTree->record(hostPosition[i], rec, EDirectionalFilter::ENearest);
        }
    }

    size_t splitThreshold = static_cast<size_t>(std::sqrt(std::pow(2, m_iteration) * 16) * 48.f);
    m_sTree->refine(splitThreshold, -1);
    
    m_sTree->forEachDTreeWrapper([](DTreeWrapper* dTree) {
        dTree->reset(6, 0.01f);
    });
    
    m_iteration++;
}

void GuidingManager::preprocess() {}

void GuidingManager::postprocess() {
    if (m_mode != PATH_GUIDING_PPG || !m_sTree) return;

    m_sTree->forEachDTreeWrapper([](DTreeWrapper* dTree) {
        dTree->build();
    });
}

void GuidingManager::destroy() {
    if (m_sTree) {
        delete m_sTree;
        m_sTree = nullptr;
    }
}

} // namespace futaba