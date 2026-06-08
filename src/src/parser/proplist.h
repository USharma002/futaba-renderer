#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <stdexcept>
#include <optional>
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

    // tryGet helpers returning std::optional
    std::optional<bool> tryGetBoolean(const std::string& name) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) return std::nullopt;
        if (auto b = std::get_if<bool>(&it->second)) return *b;
        if (auto i = std::get_if<int>(&it->second)) return *i != 0;
        if (auto f = std::get_if<float>(&it->second)) return *f != 0.f;
        return std::nullopt;
    }

    std::optional<int> tryGetInteger(const std::string& name) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) return std::nullopt;
        if (auto i = std::get_if<int>(&it->second)) return *i;
        if (auto f = std::get_if<float>(&it->second)) return (int)*f;
        if (auto b = std::get_if<bool>(&it->second)) return *b ? 1 : 0;
        return std::nullopt;
    }

    std::optional<float> tryGetFloat(const std::string& name) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) return std::nullopt;
        if (auto f = std::get_if<float>(&it->second)) return *f;
        if (auto i = std::get_if<int>(&it->second)) return (float)*i;
        if (auto b = std::get_if<bool>(&it->second)) return *b ? 1.f : 0.f;
        return std::nullopt;
    }

    std::optional<std::string> tryGetString(const std::string& name) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) return std::nullopt;
        if (auto s = std::get_if<std::string>(&it->second)) return *s;
        return std::nullopt;
    }

    std::optional<::Color3f> tryGetColor(const std::string& name) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) return std::nullopt;
        if (const ColorValue* cv = std::get_if<ColorValue>(&it->second)) return cv->value;
        if (const float* f = std::get_if<float>(&it->second)) return ::Color3f(*f, *f, *f);
        if (const int* i = std::get_if<int>(&it->second)) return ::Color3f((float)*i, (float)*i, (float)*i);
        if (const bool* b = std::get_if<bool>(&it->second)) return ::Color3f(*b ? 1.f : 0.f);
        if (const VectorValue* vv = std::get_if<VectorValue>(&it->second)) return ::Color3f(vv->value.x, vv->value.y, vv->value.z);
        if (const ::Point3f* pt = std::get_if<::Point3f>(&it->second)) return ::Color3f(pt->x, pt->y, pt->z);
        return std::nullopt;
    }

    std::optional<::Point3f> tryGetPoint(const std::string& name) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) return std::nullopt;
        if (const ::Point3f* pt = std::get_if<::Point3f>(&it->second)) return *pt;
        if (const VectorValue* vv = std::get_if<VectorValue>(&it->second)) return ::Point3f(vv->value.x, vv->value.y, vv->value.z);
        if (const ColorValue* cv = std::get_if<ColorValue>(&it->second)) return ::Point3f(cv->value.x, cv->value.y, cv->value.z);
        if (const float* f = std::get_if<float>(&it->second)) return ::Point3f(*f, *f, *f);
        if (const int* i = std::get_if<int>(&it->second)) return ::Point3f((float)*i, (float)*i, (float)*i);
        return std::nullopt;
    }

    std::optional<::Vector3f> tryGetVector(const std::string& name) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) return std::nullopt;
        if (const VectorValue* vv = std::get_if<VectorValue>(&it->second)) return vv->value;
        if (const ::Point3f* pt = std::get_if<::Point3f>(&it->second)) return ::Vector3f(pt->x, pt->y, pt->z);
        if (const ColorValue* cv = std::get_if<ColorValue>(&it->second)) return ::Vector3f(cv->value.x, cv->value.y, cv->value.z);
        if (const float* f = std::get_if<float>(&it->second)) return ::Vector3f(*f, *f, *f);
        if (const int* i = std::get_if<int>(&it->second)) return ::Vector3f((float)*i, (float)*i, (float)*i);
        return std::nullopt;
    }

    // Required getters
    bool getBoolean(const std::string& name) const {
        auto val = tryGetBoolean(name);
        if (!val) throwOrMissing(name, "boolean-coercible");
        return *val;
    }

    int getInteger(const std::string& name) const {
        auto val = tryGetInteger(name);
        if (!val) throwOrMissing(name, "integer-coercible");
        return *val;
    }

    float getFloat(const std::string& name) const {
        auto val = tryGetFloat(name);
        if (!val) throwOrMissing(name, "float-coercible");
        return *val;
    }

    std::string getString(const std::string& name) const {
        auto val = tryGetString(name);
        if (!val) throwOrMissing(name, "string");
        return *val;
    }

    ::Color3f getColor(const std::string& name) const {
        auto val = tryGetColor(name);
        if (!val) throwOrMissing(name, "color-coercible");
        return *val;
    }

    ::Point3f getPoint(const std::string& name) const {
        auto val = tryGetPoint(name);
        if (!val) throwOrMissing(name, "point-coercible");
        return *val;
    }

    ::Vector3f getVector(const std::string& name) const {
        auto val = tryGetVector(name);
        if (!val) throwOrMissing(name, "vector-coercible");
        return *val;
    }

    // Default-value getters
    bool getBoolean(const std::string& name, bool defaultValue) const {
        return tryGetBoolean(name).value_or(defaultValue);
    }

    int getInteger(const std::string& name, int defaultValue) const {
        return tryGetInteger(name).value_or(defaultValue);
    }

    float getFloat(const std::string& name, float defaultValue) const {
        return tryGetFloat(name).value_or(defaultValue);
    }

    std::string getString(const std::string& name, const std::string& defaultValue) const {
        return tryGetString(name).value_or(defaultValue);
    }

    ::Color3f getColor(const std::string& name, const ::Color3f& defaultValue) const {
        return tryGetColor(name).value_or(defaultValue);
    }

    ::Point3f getPoint(const std::string& name, const ::Point3f& defaultValue) const {
        return tryGetPoint(name).value_or(defaultValue);
    }

    ::Vector3f getVector(const std::string& name, const ::Vector3f& defaultValue) const {
        return tryGetVector(name).value_or(defaultValue);
    }

    bool hasProperty(const std::string& name) const {
        return m_properties.find(name) != m_properties.end();
    }

private:
    void throwOrMissing(const std::string& name, const char* expectedType) const {
        if (hasProperty(name)) {
            throw std::runtime_error("Property '" + name + "' has wrong type (expected " + expectedType + ")");
        } else {
            throw std::runtime_error("Property '" + name + "' is missing");
        }
    }

    std::unordered_map<std::string, Value> m_properties;
};

} // namespace futaba
