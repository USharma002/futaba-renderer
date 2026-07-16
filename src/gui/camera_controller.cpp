#include "camera_controller.h"
#include <algorithm>
#include <iostream>

namespace futaba {

static constexpr float kMinFov = 5.f;
static constexpr float kMaxFov = 120.f;

CameraController::CameraController()
    : m_camPos(0.f, 0.f, 2.5f)
    , m_camForward(0.f, 0.f, -1.f)
    , m_camUp(0.f, 1.f, 0.f)
    , m_moveSpeed(2.f)
    , m_currentFov(39.3077f)
    , m_currentFocusDistance(1.f)
    , m_currentApertureRadius(0.0f)
{
}

void CameraController::updateCamera(PerspectiveCamera& camera, int renderWidth, int renderHeight) {
    float aspect = (float)renderWidth / (float)std::max(1, renderHeight);
    camera.init(
        Point3f(m_camPos.x, m_camPos.y, m_camPos.z),
        Point3f(m_camPos.x + m_camForward.x, m_camPos.y + m_camForward.y, m_camPos.z + m_camForward.z),
        m_camUp,
        m_currentFov,
        aspect,
        m_currentFocusDistance,
        m_currentApertureRadius
    );
}

bool CameraController::handleKeyboard(int key, int action) {
    if (key >= 0 && key < 1024) {
        if (action == 1 /* GLFW_PRESS */) m_keys[key] = true;
        else if (action == 0 /* GLFW_RELEASE */) m_keys[key] = false;
        return true;
    }
    return false;
}

bool CameraController::handleMouseButton(int button, bool down) {
    if (button == 0 /* GLFW_MOUSE_BUTTON_LEFT */ || button == 1 /* GLFW_MOUSE_BUTTON_RIGHT */) {
        m_rightMousePressed = down;
        return true;
    }
    return false;
}

bool CameraController::handleMouseMotion(const nanogui::Vector2i &rel) {
    if (m_rightMousePressed) {
        float dx = -rel.x() * 0.15f;
        float dy = -rel.y() * 0.15f;

        m_camForward = normalize(Matrix4f::rotate(m_camUp, dx) * m_camForward);
        Vector3f right = normalize(cross(m_camForward, m_camUp));
        Vector3f newForward = normalize(Matrix4f::rotate(right, dy) * m_camForward);

        if (std::abs(dot(newForward, m_camUp)) < 0.99f) {
            m_camForward = newForward;
        }
        return true;
    }
    return false;
}

bool CameraController::handleScroll(const nanogui::Vector2f &rel) {
    m_currentFov -= rel.y() * 2.f;
    if (m_currentFov < kMinFov) m_currentFov = kMinFov;
    if (m_currentFov > kMaxFov) m_currentFov = kMaxFov;
    return true;
}

bool CameraController::update(float deltaTime) {
    float spd = m_moveSpeed * deltaTime;
    if (m_keys[340] /* GLFW_KEY_LEFT_SHIFT */) spd *= 3.f;

    Vector3f fwd = m_camForward;
    Vector3f up = m_camUp;
    Vector3f right = normalize(cross(fwd, up));

    bool moved = false;
    if (m_keys[87] /* W */) { m_camPos += fwd * spd; moved = true; }
    if (m_keys[83] /* S */) { m_camPos += fwd * -spd; moved = true; }
    if (m_keys[68] /* D */) { m_camPos += right * spd; moved = true; }
    if (m_keys[65] /* A */) { m_camPos += right * -spd; moved = true; }
    if (m_keys[69] /* E */) { m_camPos += up * spd; moved = true; }
    if (m_keys[81] /* Q */) { m_camPos += up * -spd; moved = true; }

    return moved;
}

void CameraController::setFromConfig(const Point3f& origin, const Point3f& target, const Vector3f& up, float fov, float focusDistance, float apertureRadius) {
    m_camPos = Vector3f(origin.x, origin.y, origin.z);
    Vector3f fwd(target.x - origin.x, target.y - origin.y, target.z - origin.z);
    m_camForward = normalize(fwd);
    m_camUp = up;
    m_currentFov = fov;
    m_currentFocusDistance = focusDistance;
    m_currentApertureRadius = apertureRadius;
}

} // namespace futaba
