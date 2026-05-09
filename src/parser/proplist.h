#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <stdexcept>
#include "types.cuh"

namespace futaba {

class PropertyList {
public:
    struct ColorValue { ::Color3f value; };
    struct VectorValue { ::Vector3f value; };

    using Value = std::variant<bool, int, float, std::string, ::Point3f, ColorValue, VectorValue>;

    void setBoolean(const std::string& name, bool value) { m_properties[name] = value; }
    void setInteger(const std::string& name, int value) { m_properties[name] = value; }
    void setFloat(const std::string& name, float value) { m_properties[name] = value; }
    void setString(const std::string& name, const std::string& value) { m_properties[name] = value; }
    void setColor(const std::string& name, const ::Color3f& value) { m_properties[name] = ColorValue{ value }; }
    void setPoint(const std::string& name, const ::Point3f& value) { m_properties[name] = value; }
    void setVector(const std::string& name, const ::Vector3f& value) { m_properties[name] = VectorValue{ value }; }

    bool getBoolean(const std::string& name) const { return getRequired<bool>(name, "boolean"); }
    int getInteger(const std::string& name) const { return getRequired<int>(name, "integer"); }
    float getFloat(const std::string& name) const { return getRequired<float>(name, "float"); }
    std::string getString(const std::string& name) const { return getRequired<std::string>(name, "string"); }
    ::Color3f getColor(const std::string& name) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) {
            throw std::runtime_error("Property '" + name + "' is missing");
        }
        const ColorValue* value = std::get_if<ColorValue>(&it->second);
        if (value == nullptr) {
            throw std::runtime_error("Property '" + name + "' has wrong type (expected <color>)");
        }
        return value->value;
    }
    ::Point3f getPoint(const std::string& name) const { return getRequired<::Point3f>(name, "point"); }
    ::Vector3f getVector(const std::string& name) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) {
            throw std::runtime_error("Property '" + name + "' is missing");
        }
        const VectorValue* value = std::get_if<VectorValue>(&it->second);
        if (value == nullptr) {
            throw std::runtime_error("Property '" + name + "' has wrong type (expected <vector>)");
        }
        return value->value;
    }

    bool getBoolean(const std::string& name, bool defaultValue) const { return getWithDefault<bool>(name, defaultValue, "boolean"); }
    int getInteger(const std::string& name, int defaultValue) const { return getWithDefault<int>(name, defaultValue, "integer"); }
    float getFloat(const std::string& name, float defaultValue) const { return getWithDefault<float>(name, defaultValue, "float"); }
    std::string getString(const std::string& name, const std::string& defaultValue) const { return getWithDefault<std::string>(name, defaultValue, "string"); }
    ::Color3f getColor(const std::string& name, const ::Color3f& defaultValue) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) {
            return defaultValue;
        }
        const ColorValue* value = std::get_if<ColorValue>(&it->second);
        if (value == nullptr) {
            throw std::runtime_error("Property '" + name + "' has wrong type (expected <color>)");
        }
        return value->value;
    }
    ::Point3f getPoint(const std::string& name, const ::Point3f& defaultValue) const { return getWithDefault<::Point3f>(name, defaultValue, "point"); }
    ::Vector3f getVector(const std::string& name, const ::Vector3f& defaultValue) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) {
            return defaultValue;
        }
        const VectorValue* value = std::get_if<VectorValue>(&it->second);
        if (value == nullptr) {
            throw std::runtime_error("Property '" + name + "' has wrong type (expected <vector>)");
        }
        return value->value;
    }

    bool hasProperty(const std::string& name) const {
        return m_properties.find(name) != m_properties.end();
    }

private:
    template <typename T>
    T getRequired(const std::string& name, const char* expectedType) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) {
            throw std::runtime_error("Property '" + name + "' is missing");
        }

        const T* value = std::get_if<T>(&it->second);
        if (value == nullptr) {
            throw std::runtime_error("Property '" + name + "' has wrong type (expected <" + std::string(expectedType) + ">)");
        }
        return *value;
    }

    template <typename T>
    T getWithDefault(const std::string& name, const T& defaultValue, const char* expectedType) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) {
            return defaultValue;
        }

        const T* value = std::get_if<T>(&it->second);
        if (value == nullptr) {
            throw std::runtime_error("Property '" + name + "' has wrong type (expected <" + std::string(expectedType) + ">)");
        }
        return *value;
    }

    std::unordered_map<std::string, Value> m_properties;
};

} // namespace futaba
