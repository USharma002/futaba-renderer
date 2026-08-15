#pragma once
#include <cuda_runtime.h>

FUTABA_NAMESPACE_BEGIN

// RAII smart pointer for CUDA device memory
template <typename T>
class CudaUniquePtr {
private:
    T* m_ptr = nullptr;

public:
    CudaUniquePtr() = default;
    explicit CudaUniquePtr(T* ptr) : m_ptr(ptr) {}

    // Destructor frees memory automatically
    ~CudaUniquePtr() {
        reset();
    }

    // Disable copy constructors to ensure unique ownership
    CudaUniquePtr(const CudaUniquePtr&) = delete;
    CudaUniquePtr& operator=(const CudaUniquePtr&) = delete;

    // Move constructors
    CudaUniquePtr(CudaUniquePtr&& other) noexcept : m_ptr(other.m_ptr) {
        other.m_ptr = nullptr;
    }

    CudaUniquePtr& operator=(CudaUniquePtr&& other) noexcept {
        if (this != &other) {
            reset();
            m_ptr = other.m_ptr;
            other.m_ptr = nullptr;
        }
        return *this;
    }

    T* get() const { return m_ptr; }

    void reset(T* ptr = nullptr) {
        if (m_ptr) {
            cudaFree(m_ptr);
        }
        m_ptr = ptr;
    }

    T* release() {
        T* temp = m_ptr;
        m_ptr = nullptr;
        return temp;
    }

    T& operator*() const { return *m_ptr; }
    T* operator->() const { return m_ptr; }
    explicit operator bool() const { return m_ptr != nullptr; }
};

FUTABA_NAMESPACE_END
