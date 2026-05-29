#pragma once

namespace futaba {

enum PathGuidingMode {
    PATH_GUIDING_NONE = 0,
    PATH_GUIDING_SD_TREE = 1, // Practical Path Guiding using SD-Trees
    PATH_GUIDING_NPM = 2      // Path Guiding using Neural Parametric Mixtures (NPM)
};

#ifndef __CUDACC__
class GuidingManager {
public:
    GuidingManager();
    ~GuidingManager();

    // Initialize spatial/directional structures
    bool init(int width, int height);

    // Handle resolution changes
    void resize(int width, int height);

    // Reset training and distributions
    void reset();

    // Perform iteration of training/updating guiding structures
    void train(int sampleCount);

    // Pre-processing and post-processing frame hooks
    void preprocess();
    void postprocess();

    // Free any allocated memory/resources
    void destroy();

    PathGuidingMode getMode() const { return m_mode; }
    void setMode(PathGuidingMode mode) { m_mode = mode; }

private:
    PathGuidingMode m_mode;
    int m_width;
    int m_height;
};
#endif

} // namespace futaba
