#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <stdexcept>
#include "types.cuh"

FUTABA_NAMESPACE_BEGIN

class PropertyList {
public:
    using Value = std::variant<bool, int, float, std::string, Point3f, Vector3f>;

    void setBoolean(const std::string& name, bool v) { m_properties[name] = v; }
    void setInteger(const std::string& name, int v)   { m_properties[name] = v; }
    void setFloat(const std::string& name, float v)   { m_properties[name] = v; }
    void setString(const std::string& name, const std::string& v) { m_properties[name] = v; }
    void setColor(const std::string& name, const Color3f& v) { m_properties[name] = v; }
    void setPoint(const std::string& name, const Point3f& v) { m_properties[name] = v; }
    void setVector(const std::string& name, const Vector3f& v) { m_properties[name] = v; }

    bool getBoolean(const std::string& name, bool def = false) const {
        auto it = m_properties.find(name);
        if (it != m_properties.end()) {
            if (auto v = std::get_if<bool>(&it->second)) return *v;
        }
        return def;
    }

    int getInteger(const std::string& name, int def = 0) const {
        auto it = m_properties.find(name);
        if (it != m_properties.end()) {
            if (auto v = std::get_if<int>(&it->second)) return *v;
        }
        return def;
    }

    float getFloat(const std::string& name, float def = 0.f) const {
        auto it = m_properties.find(name);
        if (it != m_properties.end()) {
            if (auto v = std::get_if<float>(&it->second)) return *v;
            if (auto v = std::get_if<int>(&it->second)) return (float)*v;
        }
        return def;
    }

    std::string getString(const std::string& name, const std::string& def = "") const {
        auto it = m_properties.find(name);
        if (it != m_properties.end()) {
            if (auto v = std::get_if<std::string>(&it->second)) return *v;
        }
        return def;
    }

    Color3f getColor(const std::string& name, const Color3f& def = Color3f(0.f)) const {
        auto it = m_properties.find(name);
        if (it != m_properties.end()) {
            if (auto v = std::get_if<Vector3f>(&it->second)) return *v;
            if (auto v = std::get_if<float>(&it->second)) return Color3f(*v);
            if (auto v = std::get_if<Point3f>(&it->second)) return Color3f(v->x, v->y, v->z);
        }
        return def;
    }

    Point3f getPoint(const std::string& name, const Point3f& def = Point3f(0.f)) const {
        auto it = m_properties.find(name);
        if (it != m_properties.end()) {
            if (auto v = std::get_if<Point3f>(&it->second)) return *v;
            if (auto v = std::get_if<Vector3f>(&it->second)) return Point3f(v->x, v->y, v->z);
        }
        return def;
    }

    Vector3f getVector(const std::string& name, const Vector3f& def = Vector3f(0.f)) const {
        auto it = m_properties.find(name);
        if (it != m_properties.end()) {
            if (auto v = std::get_if<Vector3f>(&it->second)) return *v;
            if (auto v = std::get_if<Point3f>(&it->second)) return Vector3f(v->x, v->y, v->z);
        }
        return def;
    }

    bool hasProperty(const std::string& name) const {
        return m_properties.find(name) != m_properties.end();
    }

    const std::unordered_map<std::string, Value>& getProperties() const {
        return m_properties;
    }

private:
    std::unordered_map<std::string, Value> m_properties;
};

FUTABA_NAMESPACE_END
