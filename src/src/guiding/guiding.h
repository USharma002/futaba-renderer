#pragma once
#include "ppg/stree.h"
#include "training_buffer.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace nanogui {
    class Widget;
}

namespace futaba {

struct LaunchParams;
class HDRFilm;

enum PathGuidingMode {
    PATH_GUIDING_NONE = 0,
    PATH_GUIDING_PPG = 1, 
    PATH_GUIDING_NPM = 2      
};

class GuidingMethod {
public:
    virtual ~GuidingMethod() = default;
    virtual std::string getName() const = 0;
    virtual int getMode() const = 0;
    virtual void initialize(int width, int height) {}
    virtual void resize(int width, int height) {}
    virtual void setSceneBounds(const AABB& bounds) {}
    virtual void reset() {}
    virtual void renderUI(nanogui::Widget* parent, std::function<void()> clearFilm, std::function<void()> trainTree) {}
    virtual void updateLaunchParams(LaunchParams& params) const {}
    virtual void preprocess() {}
    virtual void postprocess() {}
    virtual void train(const TrainingBufferManager& bufferManager, int maxDepth, const HDRFilm* film = nullptr) {}
    virtual std::vector<std::string> getVisualizerBuffers() const { return {"Active"}; }
};

using GuidingUI = GuidingMethod;

class GuidingRegistry {
public:
    static std::vector<std::shared_ptr<GuidingMethod>>& getMethods();
    static std::shared_ptr<GuidingMethod> getMethod(int mode);
    static std::vector<std::shared_ptr<GuidingMethod>>& getAlgorithms() { return getMethods(); }
    static std::shared_ptr<GuidingMethod> getAlgorithm(int mode) { return getMethod(mode); }
};

#ifndef __CUDACC__
class GuidingManager {
public:
    GuidingManager();
    ~GuidingManager();

    bool init(int width, int height);
    void resize(int width, int height);
    void setSceneBounds(const AABB& bounds);
    void reset();
    
    // Ingests data collected from TrainingBuffers and updates tree statistics
    void train(const TrainingBufferManager& bufferManager, int maxDepth, const HDRFilm* film = nullptr);

    void preprocess();
    void postprocess();
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
