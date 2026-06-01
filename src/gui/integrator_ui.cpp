#include "integrator_ui.h"
#include "path.cuh"
#include "normals.cuh"
#include "depth.cuh"
#include "albedo.cuh"
#include "phong.cuh"
#include "primitives.cuh"
#include "heatmap.cuh"
#include "volpath.cuh"

namespace futaba {

std::vector<std::shared_ptr<IntegratorUI>>& IntegratorRegistry::getIntegrators() {
    static std::vector<std::shared_ptr<IntegratorUI>> integrators;
    if (integrators.empty()) {
        integrators.push_back(std::make_shared<PathIntegratorUI>());
        integrators.push_back(std::make_shared<NormalsIntegratorUI>());
        integrators.push_back(std::make_shared<DepthIntegratorUI>());
        integrators.push_back(std::make_shared<AlbedoIntegratorUI>());
        integrators.push_back(std::make_shared<PhongIntegratorUI>());
        integrators.push_back(std::make_shared<PrimitivesIntegratorUI>());
        integrators.push_back(std::make_shared<HeatmapIntegratorUI>());
        integrators.push_back(std::make_shared<VolPathIntegratorUI>());
    }
    return integrators;
}

std::shared_ptr<IntegratorUI> IntegratorRegistry::getIntegrator(int mode) {
    for (const auto& integrator : getIntegrators()) {
        if (integrator->getMode() == mode) {
            return integrator;
        }
    }
    return nullptr;
}

} // namespace futaba
