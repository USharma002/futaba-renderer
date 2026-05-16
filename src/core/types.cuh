#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

// -----------------------------------------------------------------------------
// MACROS & UTILS
// -----------------------------------------------------------------------------
// HD = Host + Device + Inline. Forces the compiler to embed the function
// directly into the kernel, skipping function call overhead.
#define HD __host__ __device__ inline

// Fast math fallback for CPU (MSVC/GCC) vs GPU (NVCC)
#ifdef __CUDA_ARCH__
    #define FAST_RSQRT(x) rsqrtf(x)
    #define FAST_MAX(a, b) fmaxf(a, b)
    #define FAST_MIN(a, b) fminf(a, b)
#else
    #define FAST_RSQRT(x) (1.0f / std::sqrt(x))
    #define FAST_MAX(a, b) std::max(a, b)
    #define FAST_MIN(a, b) std::min(a, b)
#endif

// -----------------------------------------------------------------------------
// TEMPLATED VECTOR & POINT CLASSES
// -----------------------------------------------------------------------------

// Vector2
template <typename T>
struct Vector2 {
    T x, y;
    
    HD Vector2() : x(0), y(0) {}
    HD Vector2(T x, T y) : x(x), y(y) {}
    HD Vector2(T v) : x(v), y(v) {}

    HD Vector2 operator+(const Vector2& v) const { return Vector2(x + v.x, y + v.y); }
    HD Vector2 operator-(const Vector2& v) const { return Vector2(x - v.x, y - v.y); }
    HD Vector2 operator*(T s) const { return Vector2(x * s, y * s); }
    HD Vector2 operator/(T s) const { T inv = 1 / s; return Vector2(x * inv, y * inv); }
    
    HD Vector2& operator+=(const Vector2& v) { x += v.x; y += v.y; return *this; }
    HD Vector2& operator*=(T s) { x *= s; y *= s; return *this; }
    
    HD Vector2 operator-() const { return Vector2(-x, -y); }
    
    HD T& operator[](int i) { return (&x)[i]; }
    HD const T& operator[](int i) const { return (&x)[i]; }
};

// Point2 (Semantically distinct from Vector2)
template <typename T>
struct Point2 {
    T x, y;
    
    HD Point2() : x(0), y(0) {}
    HD Point2(T x, T y) : x(x), y(y) {}
    HD Point2(T v) : x(v), y(v) {}

    // Point + Vector = Point
    HD Point2 operator+(const Vector2<T>& v) const { return Point2(x + v.x, y + v.y); }
    // Point - Point = Vector
    HD Vector2<T> operator-(const Point2& p) const { return Vector2<T>(x - p.x, y - p.y); }
    // Point - Vector = Point
    HD Point2 operator-(const Vector2<T>& v) const { return Point2(x - v.x, y - v.y); }
    
    HD T& operator[](int i) { return (&x)[i]; }
    HD const T& operator[](int i) const { return (&x)[i]; }
};

// Vector3
template <typename T>
struct Vector3 {
    T x, y, z;

    HD Vector3() : x(0), y(0), z(0) {}
    HD Vector3(T x, T y, T z) : x(x), y(y), z(z) {}
    HD Vector3(T v) : x(v), y(v), z(v) {}

    HD Vector3 operator+(const Vector3& v) const { return Vector3(x + v.x, y + v.y, z + v.z); }
    HD Vector3 operator-(const Vector3& v) const { return Vector3(x - v.x, y - v.y, z - v.z); }
    HD Vector3 operator*(const Vector3& v) const { return Vector3(x * v.x, y * v.y, z * v.z); }
    HD Vector3 operator*(T s) const { return Vector3(x * s, y * s, z * s); }
    HD Vector3 operator/(T s) const { T inv = 1 / s; return Vector3(x * inv, y * inv, z * inv); }
    
    HD Vector3& operator+=(const Vector3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    HD Vector3& operator*=(T s) { x *= s; y *= s; z *= s; return *this; }
    HD Vector3& operator*=(const Vector3& v) { x *= v.x; y *= v.y; z *= v.z; return *this; }

    HD Vector3 operator-() const { return Vector3(-x, -y, -z); }

    HD T lengthSquared() const { return x*x + y*y + z*z; }
    HD T length() const { return std::sqrt(lengthSquared()); }
    
    HD T& operator[](int i) { return (&x)[i]; }
    HD const T& operator[](int i) const { return (&x)[i]; }
};

// Point3
template <typename T>
struct Point3 {
    T x, y, z;

    HD Point3() : x(0), y(0), z(0) {}
    HD Point3(T x, T y, T z) : x(x), y(y), z(z) {}
    HD Point3(T v) : x(v), y(v), z(v) {}

    HD Point3 operator+(const Vector3<T>& v) const { return Point3(x + v.x, y + v.y, z + v.z); }
    HD Vector3<T> operator-(const Point3& p) const { return Vector3<T>(x - p.x, y - p.y, z - p.z); }
    HD Point3 operator-(const Vector3<T>& v) const { return Point3(x - v.x, y - v.y, z - v.z); }
    
    // For affine combinations (e.g. barycentric coordinates)
    HD Point3 operator*(T s) const { return Point3(x * s, y * s, z * s); }
    HD Point3 operator+(const Point3& p) const { return Point3(x + p.x, y + p.y, z + p.z); }
    
    HD T& operator[](int i) { return (&x)[i]; }
    HD const T& operator[](int i) const { return (&x)[i]; }
};

// -----------------------------------------------------------------------------
// TYPE ALIASES (The standard Nori/PBRT types you requested)
// -----------------------------------------------------------------------------
typedef Vector2<float> Vector2f;
typedef Vector2<int>   Vector2i;
typedef Point2<float>  Point2f;
typedef Point2<int>    Point2i;

typedef Vector3<float> Vector3f;
typedef Vector3<int>   Vector3i;
typedef Point3<float>  Point3f;
typedef Point3<int>    Point3i;

// Color is mathematically identical to a Vector3f, but semantically represents RGB
typedef Vector3<float> Color3f;

// -----------------------------------------------------------------------------
// GLOBAL MATH FUNCTIONS (CUDA Optimized)
// -----------------------------------------------------------------------------

// Dot Product
template <typename T>
HD T dot(const Vector3<T>& a, const Vector3<T>& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Cross Product
template <typename T>
HD Vector3<T> cross(const Vector3<T>& a, const Vector3<T>& b) {
    return Vector3<T>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Normalize (Uses fast hardware inverse square root for floats on GPU)
HD Vector3f normalize(const Vector3f& v) {
    float invLen = FAST_RSQRT(v.x * v.x + v.y * v.y + v.z * v.z);
    return Vector3f(v.x * invLen, v.y * invLen, v.z * invLen);
}

// Left scalar multiplication: float * Vector3f
template <typename T> HD Vector3<T> operator*(T s, const Vector3<T>& v) { return v * s; }
template <typename T> HD Point3<T> operator*(T s, const Point3<T>& p) { return p * s; }

// Utility functions
HD float clamp(float val, float minVal, float maxVal) {
    return FAST_MAX(minVal, FAST_MIN(maxVal, val));
}

HD Vector3f lerp(const Vector3f& a, const Vector3f& b, float t) {
    return a * (1.0f - t) + b * t;
}

// -----------------------------------------------------------------------------
// MATRIX4x4
// -----------------------------------------------------------------------------
struct Matrix4f {
    float m[4][4];

    // Default constructor creates Identity Matrix
    HD Matrix4f() {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                m[i][j] = (i == j) ? 1.0f : 0.0f;
    }

    // Matrix Multiplication
    HD Matrix4f operator*(const Matrix4f& rhs) const {
        Matrix4f res;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                res.m[i][j] = m[i][0] * rhs.m[0][j] +
                              m[i][1] * rhs.m[1][j] +
                              m[i][2] * rhs.m[2][j] +
                              m[i][3] * rhs.m[3][j];
            }
        }
        return res;
    }

    // Multiply Vector (Ignores translation column)
    HD Vector3f operator*(const Vector3f& v) const {
        return Vector3f(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
        );
    }

    // Multiply Point (Applies translation column, performs perspective divide)
    HD Point3f operator*(const Point3f& p) const {
        float x = m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3];
        float y = m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3];
        float z = m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3];
        float w = m[3][0] * p.x + m[3][1] * p.y + m[3][2] * p.z + m[3][3];
        
        if (w == 1.0f || w == 0.0f) return Point3f(x, y, z);
        return Point3f(x / w, y / w, z / w);
    }

    // Translation Matrix
    static HD Matrix4f translate(const Vector3f& v) {
        Matrix4f m;
        m.m[0][3] = v.x;
        m.m[1][3] = v.y;
        m.m[2][3] = v.z;
        return m;
    }

    // Scale Matrix
    static HD Matrix4f scale(const Vector3f& v) {
        Matrix4f m;
        m.m[0][0] = v.x;
        m.m[1][1] = v.y;
        m.m[2][2] = v.z;
        return m;
    }

    // Rotate Matrix (angle in degrees)
    static HD Matrix4f rotate(const Vector3f& axis, float angle) {
        float rad = angle * 3.14159265358979323846f / 180.0f;
        float c = cosf(rad);
        float s = sinf(rad);
        Vector3f a = normalize(axis);
        Matrix4f m;
        m.m[0][0] = a.x * a.x + (1.0f - a.x * a.x) * c;
        m.m[0][1] = a.x * a.y * (1.0f - c) - a.z * s;
        m.m[0][2] = a.x * a.z * (1.0f - c) + a.y * s;
        
        m.m[1][0] = a.x * a.y * (1.0f - c) + a.z * s;
        m.m[1][1] = a.y * a.y + (1.0f - a.y * a.y) * c;
        m.m[1][2] = a.y * a.z * (1.0f - c) - a.x * s;
        
        m.m[2][0] = a.x * a.z * (1.0f - c) - a.y * s;
        m.m[2][1] = a.y * a.z * (1.0f - c) + a.x * s;
        m.m[2][2] = a.z * a.z + (1.0f - a.z * a.z) * c;
        return m;
    }
};