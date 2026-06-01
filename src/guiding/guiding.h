#pragma once
#include "ppg/stree.h"
#include "training_buffer.h"

namespace futaba {

enum PathGuidingMode {
    PATH_GUIDING_NONE = 0,
    PATH_GUIDING_PPG = 1, 
    PATH_GUIDING_NPM = 2      
};

#ifndef __CUDACC__
class GuidingManager {
public:
    GuidingManager();
    ~GuidingManager();

    bool init(int width, int height);
    void resize(int width, int height);
    void reset();
    
    // Ingests data collected from TrainingBuffers and updates tree statistics
    void train(const TrainingBufferManager& bufferManager, int maxDepth);

    void preprocess();
    void postprocess();
    void destroy();

    PathGuidingMode getMode() const { return m_mode; }
    void setMode(PathGuidingMode mode) { m_mode = mode; }
    
    // Pass raw view parameters down to the active device tracking kernels
    STreeNode* getDeviceNodes() const { return m_sTree ? m_sTree->m_nodes : nullptr; }
    AABB getAABB() const { return m_sTree ? m_sTree->m_aabb : AABB(); }

private:
    PathGuidingMode m_mode;
    int m_width;
    int m_height;
    
    STree* m_sTree = nullptr; // Host allocation owning device-resident pointers
    int m_iteration = 0;
};
#endif

} // namespace futaba