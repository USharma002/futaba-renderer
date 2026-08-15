#include "integrator_ui.h"

FUTABA_NAMESPACE_BEGIN

class PhongIntegratorUI : public IntegratorUI {
public:
    PhongIntegratorUI()
        : m_lightDir(1.f, 1.f, 1.f),
          m_ambient(0.12f),
          m_diffuse(0.88f),
          m_specular(0.35f),
          m_shininess(32.f) {}

    std::string getName() const override { return "Phong"; }
    int getMode() const override { return INTEGRATOR_PHONG; }

    void renderUI(nanogui::Widget* parent, std::function<void()> clearFilm) override {
        auto addPhongSlider = [parent, clearFilm](const std::string &name, float minVal,
                                                 float maxVal, float initialValue,
                                                 const std::function<void(float)> &onChange) {
            new nanogui::Label(parent, name, "sans-bold");
            nanogui::Widget *panel = new nanogui::Widget(parent);
            panel->setLayout(new nanogui::BoxLayout(nanogui::Orientation::Horizontal, nanogui::Alignment::Middle, 0, 10));

            nanogui::Slider *slider = new nanogui::Slider(panel);
            slider->setFixedWidth(120);
            
            float t = (initialValue - minVal) / (maxVal - minVal);
            slider->setValue(t < 0.f ? 0.f : (t > 1.f ? 1.f : t));

            nanogui::Label *valueLabel = new nanogui::Label(panel, std::to_string(initialValue));
            slider->setCallback([minVal, maxVal, valueLabel, onChange, clearFilm](float t) {
                const float value = minVal + t * (maxVal - minVal);
                valueLabel->setCaption(std::to_string(value));
                onChange(value);
                clearFilm();
            });
        };

        addPhongSlider("Ambient", 0.f, 2.f, m_ambient, [this](float v) { m_ambient = v; });
        addPhongSlider("Diffuse", 0.f, 2.f, m_diffuse, [this](float v) { m_diffuse = v; });
        addPhongSlider("Specular", 0.f, 2.f, m_specular, [this](float v) { m_specular = v; });
        addPhongSlider("Shininess", 1.f, 128.f, m_shininess, [this](float v) { m_shininess = v; });
    }

    void updateLaunchParams(LaunchParams& params) const override {
        params.phong.light_dir = m_lightDir;
        params.phong.ambient = m_ambient;
        params.phong.diffuse = m_diffuse;
        params.phong.specular = m_specular;
        params.phong.shininess = m_shininess;
    }

private:
    Vector3f m_lightDir;
    float m_ambient;
    float m_diffuse;
    float m_specular;
    float m_shininess;
};

std::vector<std::shared_ptr<IntegratorUI>>& IntegratorRegistry::getIntegrators() {
    static std::vector<std::shared_ptr<IntegratorUI>> integrators;
    if (integrators.empty()) {
        integrators.push_back(std::make_shared<SimpleIntegratorUI>("Path", INTEGRATOR_PATH));
        integrators.push_back(std::make_shared<SimpleIntegratorUI>("VolPath", INTEGRATOR_VOLPATH));
        integrators.push_back(std::make_shared<SimpleIntegratorUI>("Normals", INTEGRATOR_NORMALS));
        integrators.push_back(std::make_shared<SimpleIntegratorUI>("Depth", INTEGRATOR_DEPTH));
        integrators.push_back(std::make_shared<SimpleIntegratorUI>("Albedo", INTEGRATOR_ALBEDO));
        integrators.push_back(std::make_shared<PhongIntegratorUI>());
        integrators.push_back(std::make_shared<SimpleIntegratorUI>("Primitives", INTEGRATOR_PRIMITIVES));
        integrators.push_back(std::make_shared<SimpleIntegratorUI>("Heatmap", INTEGRATOR_HEATMAP));
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

FUTABA_NAMESPACE_END
