# Futaba Renderer

Futaba is a high-performance, learning-oriented physically-based renderer written in **C++ and CUDA**. Inspired by [Mitsuba](https://www.mitsuba-renderer.org/), this project focuses on clear architecture and the implementation of advanced rendering techniques.

![Renderer Preview](assets/dragon-cbox.png)

## Project Goals

- **Educational**: Understand the internals of physically-based rendering (PBR) from first principles.
- **GPU Performance**: Utilize NVIDIA CUDA for high-performance ray tracing and Monte Carlo integration.
- **Path Guiding**: Implement advanced sampling techniques to improve convergence and reduce noise.
- **Modular Architecture**: Clean separation of concerns (integrators, shapes, materials, sensors, etc.).

## Features

### Current Implementation
- **GPU Acceleration**: Entire rendering pipeline implemented in **CUDA** for massive parallelism. Incorporates **NVIDIA OptiX** for hardware-accelerated ray-geometry intersections alongside a highly optimized software BVH fallback.
- **Interactive UI**: Real-time viewport driven by NanoGUI featuring:
  ![Interactive UI](assets/futaba-window.png)
  - Robust WASD navigation utilizing quaternion-style relative axis-angle rotations, fully supporting arbitrary scene 'up' vectors without gimbal lock.
  - Dynamic GPU state toggles for **Anti-aliasing** (subpixel stochastic jittering) and **Smooth Shading** (interpolated vertex normals vs. flat face normals).
  - An orientation gizmo and interactive FOV/depth sliders that instantly trigger accumulation buffer resets.
- **Scene Parsing**: 
  - Comprehensive custom XML loader mirroring the Mitsuba architecture.
  - Deep support for composite `<transform>` hierarchies (`scale`, `rotate`, `translate`).
  - Advanced OBJ parsing supporting `v/vt/vn` indices and pre-transforming geometric normals.
- **Integrators**: 
  - **Path Tracing**: Full Monte Carlo integration with Russian Roulette.
    ![Path Tracing](assets/dragon-cbox.png)
  - **Normals**: Surface normal visualization for debugging smoothing groups and face orientations.
    ![Surface Normals](assets/dragon-cbox-normals.png)
  - **Heatmap**: Visualization of AABB intersection complexity, normalized to a dynamic color scale.
    ![Intersection Heatmap](assets/dragon-cbox-heatmap.png)
- **Films**: 32-bit HDR accumulation mapped directly to zero-copy OpenGL PBOs with **EXR** export support.

### Planned Features
- **Advanced Path Guiding**: Learn and adapt to scene-specific light transport patterns.
- **Complex Materials**: Rough conductors, dielectrics, and layered materials.
- **Media**: Volumetric rendering support.

## Architecture Overview

### Rendering Pipeline

1. **Sensor**: Generates primary rays on the GPU based on camera orientation, optionally applying subpixel stochastic jitter for anti-aliasing.
2. **Ray Tracing**: Dispatches rays to **NVIDIA OptiX** pipeline modules (`__raygen__`, `__closesthit__`, `__miss__`), utilizing hardware RT cores for geometry queries.
3. **Integrator**: Computes radiance using CUDA kernels, resolving varying material types and debugging modes directly on the GPU.
4. **BSDF**: Evaluates physically-based material properties and samples indirect lighting paths.
5. **Film**: Accumulates floating-point samples directly into mapped OpenGL Pixel Buffer Objects (PBOs) for zero-copy, real-time screen display, while supporting CPU-side extraction for EXR archival.

## Building and Running

### Prerequisites
- **CUDA Toolkit**
- **CMake** (3.15+)
- **C++17** compatible compiler (MSVC 2019+, GCC, or Clang)

### Build
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## References

- [Mitsuba Renderer Documentation](https://www.mitsuba-renderer.org/)
- "Physically Based Rendering: From Theory to Implementation" by Pharr, Jakob, and Humphreys.
- [TinyEXR](https://github.com/syoyo/tinyexr) for HDR image I/O.

## License

This project is created for educational purposes.