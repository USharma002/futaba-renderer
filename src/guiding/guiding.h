#pragma once

namespace futaba {

enum PathGuidingMode {
    PATH_GUIDING_NONE = 0,
    PATH_GUIDING_SD_TREE = 1, // Practical Path Guiding using SD-Trees
    PATH_GUIDING_VMM = 2      // Path Guiding using Variational Mixture Models (e.g. vMM)
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

    // Perform an iteration of training/updating the guiding structures
    void train(int sampleCount);

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
