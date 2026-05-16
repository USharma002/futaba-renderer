# Futaba Renderer

Futaba is a high-performance, learning-oriented physically-based renderer written in **C++ and CUDA**. Inspired by [Mitsuba](https://www.mitsuba-renderer.org/), this project focuses on clear architecture and the implementation of advanced rendering techniques.

![Renderer Preview](assets/dragon-cbox.png)

## Available Visualizations

| Path Tracing | Albedo | Normals |
|:---:|:---:|:---:|
| ![Path Tracing](assets/modes/chess-path.png) | ![Albedo](assets/modes/chess-albedo.png) | ![Normals](assets/modes/chess-normals.png) |
| **Depth** | **Heatmap** | **Phong** |
| ![Depth](assets/modes/chess-depth.png) | ![Heatmap](assets/modes/chess-heatmap.png) | ![Phong](assets/modes/chess-phong.png) |
| **Primitives** | | |
| ![Primitives](assets/modes/chess-primitives.png) | | |

## Project Goals

- **Educational**: Understand the internals of physically-based rendering (PBR) from first principles.
- **GPU Performance**: Utilize NVIDIA CUDA for high-performance ray tracing and Monte Carlo integration.
- **Path Guiding**: Implement advanced sampling techniques to improve convergence and reduce noise.
- **Modular Architecture**: Clean separation of concerns (integrators, shapes, materials, sensors, etc.).

## Features

### Current Implementation
- [x] **GPU Acceleration**: CUDA-based pipeline with NVIDIA OptiX hardware acceleration and an optimized software BVH fallback.
- [x] **Interactive UI**: Real-time viewport driven by NanoGUI featuring:
   ![Interactive UI - Default](assets/futaba-window.png)
  - Smooth WASD navigation with gimbal-lock-free quaternion rotations.
  - On-screen orientation gizmo anchored to the top-right corner.
  - Responsive, non-distorting viewport that dynamically adapts to window resizing.
  - GPU toggles for Anti-aliasing and Smooth Shading.
  - Interactive FOV and depth sliders with real-time accumulation reset.

  ![Interactive UI - Spaceship](assets/spaceship-window.png)
- [x] **Scene Parsing**: 
  - Loading logic based on the **Nori** renderer, with an overall structure based on a **Mitsuba hybrid** approach.
  - XML loader supporting nested `<transform>` blocks and advanced OBJ parsing.
- [x] **Integrators**: 
  - [x] **Path Tracing**: Full Monte Carlo integration with Russian Roulette.
    ![Path Tracing](assets/dragon-cbox.png)
  - [x] **Normals**: Surface normal visualization for debugging.
    ![Surface Normals](assets/dragon-cbox-normals.png)
  - [x] **Heatmap**: AABB intersection complexity visualization.
    ![Intersection Heatmap](assets/dragon-cbox-heatmap.png)
- [x] **Films**: 32-bit HDR accumulation with zero-copy OpenGL PBO display and EXR export support.

### Planned Features
- [x] Done
  - [x] Normal Visualization
  - [x] Path tracing
- [~] Partially done
  - [~] Various Materials
  - [~] Textures and Environment map support
- [ ] Not started
  - [ ] Next Event Estimation (NEE)
  - [ ] Multiple Importance Sampling (MIS)
  - [ ] Path Guiding (PPG, SDMM, NPM etc.)
  - [ ] Bidirectional Path Tracing
  - [ ] Photon Mapping
  - [ ] Radiosity
  - [ ] Volume Rendering
  - [ ] Basic Differentiable Rendering

## Architecture Overview

### Rendering Pipeline

1. **Sensor**: Generates primary rays on the GPU based on camera orientation, optionally applying subpixel stochastic jitter for anti-aliasing.
2. **Ray Tracing**: Dispatches rays to **NVIDIA OptiX** pipeline modules (`__raygen__`, `__closesthit__`, `__miss__`), utilizing hardware RT cores for geometry queries.
3. **Integrator**: Computes radiance using CUDA kernels, resolving varying material types and debugging modes directly on the GPU.
4. **BSDF**: Evaluates physically-based material properties and samples indirect lighting paths.
5. **Film**: Accumulates floating-point samples directly into mapped OpenGL Pixel Buffer Objects (PBOs) for zero-copy, real-time screen display, while supporting CPU-side extraction for EXR archival.


```mermaid

flowchart TD

  

%% Styling

classDef cpu fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000;

classDef gpu fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000;

classDef data fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000;

  

%% Nodes

main[main.cpp]:::cpu

FS[FutabaScreen]:::cpu

SL[SceneLoader]:::cpu

LS[LoadedScene CPU]:::data

RH[RendererHost]:::cpu

  

OPTIX[OptixPipeline]:::cpu

KERNEL[CUDA Render Kernel]:::gpu

SGPU[Scene GPU]:::gpu

  

SAMP[Sampler]:::gpu

CAM[PerspectiveCamera]:::gpu

INTG[Integrator]:::gpu

FILM[HDRFilm]:::data

  

BVH[BVH / Nodes]:::data

TRIS[Geometry]:::data

MATS[Materials]:::data

EMITS[Emitters]:::data

  

%% CPU / App Flow

main --> FS

FS -->|Loads XML| SL

SL -->|Builds| LS

LS -->|Passes to| RH

FS -->|Drives| RH

  

%% CPU to GPU Boundary

RH -->|Configures| OPTIX

RH -->|Allocates| SGPU

RH -->|Dispatches| KERNEL

  

%% GPU Data Structure

SGPU --> BVH

SGPU --> TRIS

SGPU --> MATS

SGPU --> EMITS

  

%% GPU Execution

KERNEL --> SAMP

KERNEL --> CAM

KERNEL --> INTG

  

INTG -->|Query & Shade| SGPU

  

%% Output

KERNEL -->|Accumulate| FILM

FILM -.->|Display Texture| FS

```

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