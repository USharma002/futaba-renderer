#pragma once

#include <nanogui/nanogui.h>
#include "perspective.cuh"

namespace futaba {

class CameraController {
public:
    CameraController();

    void updateCamera(PerspectiveCamera& camera, int renderWidth, int renderHeight);

    bool handleKeyboard(int key, int action);
    bool handleMouseButton(int button, bool down);
    bool handleMouseMotion(const nanogui::Vector2i &rel);
    bool handleScroll(const nanogui::Vector2f &rel);
    
    // Updates position based on keyboard state. Returns true if camera moved.
    bool update(float deltaTime);

    void setFromConfig(const Point3f& origin, const Point3f& target, const Vector3f& up, float fov, float focusDistance, float apertureRadius);

    float fov() const { return m_currentFov; }
    void setFov(float fov) { m_currentFov = fov; }
    
    float focusDistance() const { return m_currentFocusDistance; }
    void setFocusDistance(float d) { m_currentFocusDistance = d; }
    
    float apertureRadius() const { return m_currentApertureRadius; }
    void setApertureRadius(float a) { m_currentApertureRadius = a; }

    Vector3f position() const { return m_camPos; }
    Vector3f forward() const { return m_camForward; }
    Vector3f up() const { return m_camUp; }

private:
    bool m_keys[1024] = {false};
    bool m_rightMousePressed = false;

    Vector3f m_camPos;
    Vector3f m_camForward;
    Vector3f m_camUp;
    float m_moveSpeed;
    float m_currentFov;
    float m_currentFocusDistance;
    float m_currentApertureRadius;
};

} // namespace futaba
