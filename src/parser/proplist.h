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

    bool getBoolean(const std::string& name) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) {
            throw std::runtime_error("Property '" + name + "' is missing");
        }
        if (auto b = std::get_if<bool>(&it->second)) return *b;
        if (auto i = std::get_if<int>(&it->second)) return *i != 0;
        if (auto f = std::get_if<float>(&it->second)) return *f != 0.f;
        throw std::runtime_error("Property '" + name + "' has wrong type (expected boolean-coercible)");
    }
    int getInteger(const std::string& name) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) {
            throw std::runtime_error("Property '" + name + "' is missing");
        }
        if (auto i = std::get_if<int>(&it->second)) return *i;
        if (auto f = std::get_if<float>(&it->second)) return (int)*f;
        if (auto b = std::get_if<bool>(&it->second)) return *b ? 1 : 0;
        throw std::runtime_error("Property '" + name + "' has wrong type (expected integer-coercible)");
    }
    float getFloat(const std::string& name) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) {
            throw std::runtime_error("Property '" + name + "' is missing");
        }
        if (auto f = std::get_if<float>(&it->second)) return *f;
        if (auto i = std::get_if<int>(&it->second)) return (float)*i;
        if (auto b = std::get_if<bool>(&it->second)) return *b ? 1.f : 0.f;
        throw std::runtime_error("Property '" + name + "' has wrong type (expected float-coercible)");
    }
    std::string getString(const std::string& name) const { return getRequired<std::string>(name, "string"); }
    ::Color3f getColor(const std::string& name) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) {
            throw std::runtime_error("Property '" + name + "' is missing");
        }
        if (const ColorValue* cv = std::get_if<ColorValue>(&it->second)) return cv->value;
        if (const float* f = std::get_if<float>(&it->second)) return ::Color3f(*f, *f, *f);
        if (const int* i = std::get_if<int>(&it->second)) return ::Color3f((float)*i, (float)*i, (float)*i);
        if (const bool* b = std::get_if<bool>(&it->second)) return ::Color3f(*b ? 1.f : 0.f);
        if (const VectorValue* vv = std::get_if<VectorValue>(&it->second)) return ::Color3f(vv->value.x, vv->value.y, vv->value.z);
        if (const ::Point3f* pt = std::get_if<::Point3f>(&it->second)) return ::Color3f(pt->x, pt->y, pt->z);
        throw std::runtime_error("Property '" + name + "' has wrong type (expected color-coercible)");
    }
    ::Point3f getPoint(const std::string& name) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) {
            throw std::runtime_error("Property '" + name + "' is missing");
        }
        if (const ::Point3f* pt = std::get_if<::Point3f>(&it->second)) return *pt;
        if (const VectorValue* vv = std::get_if<VectorValue>(&it->second)) return ::Point3f(vv->value.x, vv->value.y, vv->value.z);
        if (const ColorValue* cv = std::get_if<ColorValue>(&it->second)) return ::Point3f(cv->value.x, cv->value.y, cv->value.z);
        if (const float* f = std::get_if<float>(&it->second)) return ::Point3f(*f, *f, *f);
        if (const int* i = std::get_if<int>(&it->second)) return ::Point3f((float)*i, (float)*i, (float)*i);
        throw std::runtime_error("Property '" + name + "' has wrong type (expected point-coercible)");
    }
    ::Vector3f getVector(const std::string& name) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) {
            throw std::runtime_error("Property '" + name + "' is missing");
        }
        if (const VectorValue* vv = std::get_if<VectorValue>(&it->second)) return vv->value;
        if (const ::Point3f* pt = std::get_if<::Point3f>(&it->second)) return ::Vector3f(pt->x, pt->y, pt->z);
        if (const ColorValue* cv = std::get_if<ColorValue>(&it->second)) return ::Vector3f(cv->value.x, cv->value.y, cv->value.z);
        if (const float* f = std::get_if<float>(&it->second)) return ::Vector3f(*f, *f, *f);
        if (const int* i = std::get_if<int>(&it->second)) return ::Vector3f((float)*i, (float)*i, (float)*i);
        throw std::runtime_error("Property '" + name + "' has wrong type (expected vector-coercible)");
    }

    bool getBoolean(const std::string& name, bool defaultValue) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) return defaultValue;
        if (auto b = std::get_if<bool>(&it->second)) return *b;
        if (auto i = std::get_if<int>(&it->second)) return *i != 0;
        if (auto f = std::get_if<float>(&it->second)) return *f != 0.f;
        return defaultValue;
    }
    int getInteger(const std::string& name, int defaultValue) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) return defaultValue;
        if (auto i = std::get_if<int>(&it->second)) return *i;
        if (auto f = std::get_if<float>(&it->second)) return (int)*f;
        if (auto b = std::get_if<bool>(&it->second)) return *b ? 1 : 0;
        return defaultValue;
    }
    float getFloat(const std::string& name, float defaultValue) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) return defaultValue;
        if (auto f = std::get_if<float>(&it->second)) return *f;
        if (auto i = std::get_if<int>(&it->second)) return (float)*i;
        if (auto b = std::get_if<bool>(&it->second)) return *b ? 1.f : 0.f;
        return defaultValue;
    }
    std::string getString(const std::string& name, const std::string& defaultValue) const { return getWithDefault<std::string>(name, defaultValue, "string"); }
    ::Color3f getColor(const std::string& name, const ::Color3f& defaultValue) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) return defaultValue;
        if (const ColorValue* cv = std::get_if<ColorValue>(&it->second)) return cv->value;
        if (const float* f = std::get_if<float>(&it->second)) return ::Color3f(*f, *f, *f);
        if (const int* i = std::get_if<int>(&it->second)) return ::Color3f((float)*i, (float)*i, (float)*i);
        if (const bool* b = std::get_if<bool>(&it->second)) return ::Color3f(*b ? 1.f : 0.f);
        if (const VectorValue* vv = std::get_if<VectorValue>(&it->second)) return ::Color3f(vv->value.x, vv->value.y, vv->value.z);
        if (const ::Point3f* pt = std::get_if<::Point3f>(&it->second)) return ::Color3f(pt->x, pt->y, pt->z);
        return defaultValue;
    }
    ::Point3f getPoint(const std::string& name, const ::Point3f& defaultValue) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) return defaultValue;
        if (const ::Point3f* pt = std::get_if<::Point3f>(&it->second)) return *pt;
        if (const VectorValue* vv = std::get_if<VectorValue>(&it->second)) return ::Point3f(vv->value.x, vv->value.y, vv->value.z);
        if (const ColorValue* cv = std::get_if<ColorValue>(&it->second)) return ::Point3f(cv->value.x, cv->value.y, cv->value.z);
        if (const float* f = std::get_if<float>(&it->second)) return ::Point3f(*f, *f, *f);
        if (const int* i = std::get_if<int>(&it->second)) return ::Point3f((float)*i, (float)*i, (float)*i);
        return defaultValue;
    }
    ::Vector3f getVector(const std::string& name, const ::Vector3f& defaultValue) const {
        auto it = m_properties.find(name);
        if (it == m_properties.end()) return defaultValue;
        if (const VectorValue* vv = std::get_if<VectorValue>(&it->second)) return vv->value;
        if (const ::Point3f* pt = std::get_if<::Point3f>(&it->second)) return ::Vector3f(pt->x, pt->y, pt->z);
        if (const ColorValue* cv = std::get_if<ColorValue>(&it->second)) return ::Vector3f(cv->value.x, cv->value.y, cv->value.z);
        if (const float* f = std::get_if<float>(&it->second)) return ::Vector3f(*f, *f, *f);
        if (const int* i = std::get_if<int>(&it->second)) return ::Vector3f((float)*i, (float)*i, (float)*i);
        return defaultValue;
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
