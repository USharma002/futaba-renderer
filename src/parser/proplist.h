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

    void setBoolean(const std::string& name, bool value) { 
        m_properties[name] = value; 
    }

    void setInteger(const std::string& name, int value){ 
        m_properties[name] = value; 
    }

    void setFloat(const std::string& name, float value){ 
        m_properties[name] = value; 
    }

    void setString(const std::string& name, const std::string& value){ 
        m_properties[name] = value; 
    }

    void setColor(const std::string& name, const Color3f& value){ 
        m_properties[name] = value; 
    }

    void setPoint(const std::string& name, const Point3f& value){ 
        m_properties[name] = value; 
    }

    void setVector(const std::string& name, const Vector3f& value){ 
        m_properties[name] = value; 
    }

    bool getBoolean(const std::string& name) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) throwMissing(name);
        if (auto val = std::get_if<bool>(&it->second)) return *val;
        throwType(name, "boolean");
    }

    int getInteger(const std::string& name) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) throwMissing(name);
        if (auto val = std::get_if<int>(&it->second)) return *val;
        throwType(name, "integer");
    }

    float getFloat(const std::string& name) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) throwMissing(name);
        if (auto val = std::get_if<float>(&it->second)) return *val;
        if (auto val = std::get_if<int>(&it->second)) return (float)*val;
        throwType(name, "float");
    }

    std::string getString(const std::string& name) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) throwMissing(name);
        if (auto val = std::get_if<std::string>(&it->second)) return *val;
        throwType(name, "string");
    }

    Color3f getColor(const std::string& name) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) throwMissing(name);
        if (auto val = std::get_if<Vector3f>(&it->second)) return *val;
        if (auto val = std::get_if<float>(&it->second)) return Color3f(*val);
        if (auto val = std::get_if<Point3f>(&it->second)) return Color3f(val->x, val->y, val->z);
        throwType(name, "color");
    }

    Point3f getPoint(const std::string& name) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) throwMissing(name);
        if (auto val = std::get_if<Point3f>(&it->second)) return *val;
        if (auto val = std::get_if<Vector3f>(&it->second)) return Point3f(val->x, val->y, val->z);
        throwType(name, "point");
    }

    Vector3f getVector(const std::string& name) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) throwMissing(name);
        if (auto val = std::get_if<Vector3f>(&it->second)) return *val;
        if (auto val = std::get_if<Point3f>(&it->second)) return Vector3f(val->x, val->y, val->z);
        throwType(name, "vector");
    }

    // Default value getters
    bool getBoolean(const std::string& name, bool defaultValue) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) return defaultValue;
        if (auto val = std::get_if<bool>(&it->second)) return *val;
        return defaultValue;
    }

    int getInteger(const std::string& name, int defaultValue) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) return defaultValue;
        if (auto val = std::get_if<int>(&it->second)) return *val;
        return defaultValue;
    }

    float getFloat(const std::string& name, float defaultValue) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) return defaultValue;
        if (auto val = std::get_if<float>(&it->second)) return *val;
        if (auto val = std::get_if<int>(&it->second)) return (float)*val;
        return defaultValue;
    }

    std::string getString(const std::string& name, const std::string& defaultValue) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) return defaultValue;
        if (auto val = std::get_if<std::string>(&it->second)) return *val;
        return defaultValue;
    }

    Color3f getColor(const std::string& name, const Color3f& defaultValue) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) return defaultValue;
        if (auto val = std::get_if<Vector3f>(&it->second)) return *val;
        if (auto val = std::get_if<float>(&it->second)) return Color3f(*val);
        if (auto val = std::get_if<Point3f>(&it->second)) return Color3f(val->x, val->y, val->z);
        return defaultValue;
    }

    Point3f getPoint(const std::string& name, const Point3f& defaultValue) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) return defaultValue;
        if (auto val = std::get_if<Point3f>(&it->second)) return *val;
        if (auto val = std::get_if<Vector3f>(&it->second)) return Point3f(val->x, val->y, val->z);
        return defaultValue;
    }

    Vector3f getVector(const std::string& name, const Vector3f& defaultValue) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) return defaultValue;
        if (auto val = std::get_if<Vector3f>(&it->second)) return *val;
        if (auto val = std::get_if<Point3f>(&it->second)) return Vector3f(val->x, val->y, val->z);
        return defaultValue;
    }

    bool hasProperty(const std::string& name) const {
        return m_properties.find(name) != m_properties.end();
    }

private:
    void throwMissing(const std::string& name) const {
        throw std::runtime_error("Property '" + name + "' is missing");
    }
    void throwType(const std::string& name, const std::string& type) const {
        throw std::runtime_error("Property '" + name + "' has wrong type (expected " + type + ")");
    }

    std::unordered_map<std::string, Value> m_properties;
};

FUTABA_NAMESPACE_END
