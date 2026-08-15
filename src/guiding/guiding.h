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

FUTABA_NAMESPACE_BEGIN

struct LaunchParams;

enum PathGuidingMode {
    PATH_GUIDING_NONE = 0,
    PATH_GUIDING_PPG  = 1,
};

// A path-guiding algorithm. The distribution is built on the CPU (train) and
// sampled on the GPU (via GuidingParams filled by updateLaunchParams).
class GuidingMethod {
public:
    virtual ~GuidingMethod() = default;

    virtual std::string     name() const = 0;
    virtual PathGuidingMode mode() const = 0;

    virtual void setSceneBounds(const AABB& /*bounds*/) {}
    virtual void reset() {}

    // Build the distribution from one captured training pass.
    virtual void train(const TrainingBufferManager& /*buffers*/, int /*maxDepth*/) {}

    // Fill params.guiding so the GPU integrator can sample the distribution.
    virtual void updateLaunchParams(LaunchParams& /*params*/) const {}

    virtual bool isTrainingEnabled() const { return false; }
    virtual void setTrainingEnabled(bool /*enabled*/) {}

    // Method-specific controls, drawn under the method dropdown.
    virtual void renderUI(nanogui::Widget* /*parent*/,
                          std::function<void()> /*clearFilm*/,
                          std::function<void()> /*requestTrain*/) {}
};

// Owns the available methods and the current selection. The GUI talks only to
// this facade; it forwards everything to the active method.
class GuidingManager {
public:
    GuidingManager();

    void setSceneBounds(const AABB& bounds);
    void reset();
    void train(const TrainingBufferManager& buffers, int maxDepth);
    void updateLaunchParams(LaunchParams& params) const;

    // The method dropdown is added to `grid` (a 2-column settings grid so it
    // lines up with the other rows); per-method controls are added to `panel`.
    void renderUI(nanogui::Widget* grid,
                  nanogui::Widget* panel,
                  std::function<void()> clearFilm,
                  std::function<void()> requestTrain);

    PathGuidingMode mode() const { return m_mode; }
    bool enabled() const { return m_mode != PATH_GUIDING_NONE; }
    bool isTrainingEnabled() const { return enabled() && active()->isTrainingEnabled(); }
    void setTrainingEnabled(bool e) { if (enabled()) active()->setTrainingEnabled(e); }

private:
    GuidingMethod* active() const;

    std::vector<std::unique_ptr<GuidingMethod>> m_methods;
    PathGuidingMode m_mode = PATH_GUIDING_NONE;
};

FUTABA_NAMESPACE_END
