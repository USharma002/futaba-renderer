#pragma once
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <nanogui/nanogui.h>
#include "launch_params.h"

FUTABA_NAMESPACE_BEGIN

class IntegratorUI {
public:
    virtual ~IntegratorUI() = default;
    virtual std::string getName() const = 0;
    virtual int getMode() const = 0;
    virtual void initialize() {}
    virtual void renderUI(nanogui::Widget* parent, std::function<void()> clearFilm) {}
    virtual void updateLaunchParams(LaunchParams& params) const {}
};

class IntegratorRegistry {
public:
    static std::vector<std::shared_ptr<IntegratorUI>>& getIntegrators();
    static std::shared_ptr<IntegratorUI> getIntegrator(int mode);
};

FUTABA_NAMESPACE_END
