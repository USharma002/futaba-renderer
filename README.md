# Futaba Renderer

Futaba is a high-performance, learning-oriented physically-based renderer written in **C++ and CUDA**. Inspired by [Mitsuba](https://www.mitsuba-renderer.org/), this project focuses on clear architecture and the implementation of advanced rendering techniques.

![Renderer Preview](assets/chess.png)

## High Quality Renders

| Dragon Cornell Box | Depth of Field |
|:---:|:---:|
| ![Dragon CBox](assets/dragon-cbox.png) | ![Chess DOF](assets/chess-dof.png) |
| **Volumetric Cornell Box** | **Veach MIS** |
| ![Volumetric Cornell Box](assets/cbox-volume.png) | ![Veach MIS](assets/veach-mis.png) |
| **Ford Mustang GT3** | **Teapot Filled** |
| ![Ford Mustang GT3](assets/mustang.png) | ![Teapot Filled](assets/teapot-full.png) |


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
   ![Interactive UI - Default](assets/glass-of-water-window.png)
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
  - [x] **Volumetric Path Tracing**: Radiative transport in homogeneous participating media with Henyey-Greenstein (HG) & Isotropic phase functions.
  - [x] **Next Event Estimation**: Direct-lighting support for emissive geometry and environment lighting with Multiple Importance Sampling (MIS).
  - [x] **Normals**: Surface normal visualization for debugging.
    ![Surface Normals](assets/dragon-cbox-normals.png)
  - [x] **Heatmap**: AABB intersection complexity visualization.
    ![Intersection Heatmap](assets/dragon-cbox-heatmap.png)
- [x] **Materials & Textures**:
  - Full BSDF suite: Diffuse, Dielectric, Thin Dielectric, Rough Conductor, Rough Dielectric, Rough Plastic (Beckmann microfacet model).
  - Image textures with sRGB-to-linear conversion.
  - Tangent-space **Normal Mapping** & **Bump Mapping** with per-triangle UV derivative frame derivation.
- [x] **Path Guiding**: Practical Path Guiding (PPG) using adaptive spatial-directional quadtrees (SD-trees).
- [x] **Films**: 32-bit HDR accumulation with zero-copy OpenGL PBO display and EXR export support.
- [x] **Denoising**: Optional OptiX AI denoiser with albedo/normal guide buffers and tonemapped output.


#### Denoising Demo

![Optix Denoiser Preview](assets/optix_denoiser.gif)

---

## Roadmap

- [x] Path tracing with Russian Roulette
- [x] Next Event Estimation with MIS
- [x] Volumetric path tracing (homogeneous media, HG phase function)
- [x] Environment map emitter with importance sampling
- [x] Thin-lens depth of field
- [x] Image textures on materials
- [x] Rough conductor / dielectric / plastic BSDFs (Beckmann microfacet)
- [x] OptiX AI denoising with guide buffers
- [x] Path Guiding (PPG with SD-trees)
- [x] Bump / normal mapping
- [ ] Bidirectional path tracing
- [ ] Photon mapping
- [ ] Disney principled BSDF
- [ ] Spectral rendering

---

## Architecture Overview

### Rendering Pipeline

1. **Sensor** --- generates primary rays on the GPU from the perspective camera, with optional thin-lens DoF and subpixel jitter for anti-aliasing
2. **OptiX Ray Tracing** --- dispatches rays through `__raygen__` / `__closesthit__` / `__miss__` programs using hardware RT cores
3. **Integrator** --- CUDA kernel computes radiance; selects path tracer or visualization mode
4. **BSDF** --- GPU-side material dispatch evaluates and samples the appropriate lobe (diffuse, microfacet, dielectric, etc.)
5. **Emitter Sampling** --- NEE samples area lights, point/directional lights, and environment map; MIS weights combine with BSDF sample
6. **Film** --- samples accumulate into a 32-bit float PBO for zero-copy OpenGL display; denoiser reads albedo/normal guide buffers and writes tonemapped output

```mermaid
flowchart TD

    subgraph Host ["Host (CPU)"]
        direction TB
        main["main.cpp"] --> FS["FutabaScreen (NanoGUI Viewport)"]
        FS -->|"Load XML"| SL["SceneLoader"]
        SL -->|"Build Scene"| LS["LoadedScene"]
        LS -->|"Initialize"| RH["RendererHost"]
        FS -->|"Interactive Controls"| RH
    end

    subgraph Device ["Device Pipeline (CUDA / OptiX)"]
        direction TB
        RH -->|"Configure"| OPTIX["OptiX Acceleration Structure"]
        RH -->|"Upload"| SCENE["Scene Resources (Meshes / Materials / Media)"]
        RH -->|"Launch"| KERNEL["CUDA Render Kernel"]

        KERNEL --> CAM["Camera & Sampler"]
        KERNEL --> INTG["Integrator (Path / VolPath)"]

        INTG <-->|"Hardware RT Cores"| OPTIX
        INTG -->|"Evaluate BSDF & Phase"| MATS["BSDFs / Media / Emitters"]
        SCENE -.-> MATS
    end

    subgraph Output ["Film & Post-Processing"]
        direction TB
        INTG -->|"Write Albedo & Normals"| REC["PathRecorder (Guide Buffers)"]
        KERNEL -->|"Accumulate Radiance"| FILM["32-bit HDR Film (PBO)"]
        
        REC -->|"Guide Buffers"| DENOISE["OptiX AI Denoiser"]
        FS -->|"Trigger Denoise"| DENOISE
        DENOISE -->|"Denoised Frame"| FILM
        FILM -.->|"Zero-Copy OpenGL Display"| FS
    end

    classDef cpu fill:#e3f2fd,stroke:#1565c0,stroke-width:1.5px,color:#0d47a1;
    classDef gpu fill:#e8f5e9,stroke:#2e7d32,stroke-width:1.5px,color:#1b5e20;
    classDef data fill:#fff3e0,stroke:#e65100,stroke-width:1.5px,color:#bf360c;

    class main,FS,SL,RH cpu;
    class OPTIX,KERNEL,CAM,INTG,MATS,DENOISE gpu;
    class LS,SCENE,REC,FILM data;
```

## Building and Running

### Prerequisites
- **CUDA Toolkit**
- **NVIDIA OptiX SDK** (for OptiX acceleration and the AI denoiser)
- **CMake** (3.15+)
- **C++17** compatible compiler (MSVC 2019+, GCC, or Clang)

### Build
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## References & Credits

### 3D Models & Test Scenes
- **Ford Mustang GT3**: Model created by [vecarz.com](https://sketchfab.com/3d-models/ford-mustang-gt3-wwwvecarzcom-061487e4e4fa4cd1bf7e15eae2e238ba) on Sketchfab (CC Attribution).
- **Stanford 3D Scanning Repository**: [Stanford Computer Graphics Laboratory](https://graphics.stanford.edu/data/3Dscanrep/) (Stanford Dragon, Stanford Bunny).
- **Rendering Resources & Scenes**: [Benedikt Bitterli's Rendering Resources](https://benedikt-bitterli.me/resources/) (Cornell Box, Glass of Water, Spaceship, Teapot, etc.).
- **Veach MIS**: Eric Veach (1997), [*Robust Monte Carlo Methods for Light Transport Simulation*](https://graphics.stanford.edu/papers/veach_thesis/).
- **The Cornell Box**: [Cornell University Program of Computer Graphics](https://www.graphics.cornell.edu/online/box/).
- **Utah Teapot**: Martin Newell (1975).

### Literature & Frameworks
- Müller et al., [*Practical Path Guiding for Efficient Light-Transport Simulation*](https://tom94.net/pages/publications/mueller17practical-erratum), EGSR 2017
- Pharr, Jakob, Humphreys --- *Physically Based Rendering: From Theory to Implementation* (PBRT)
- [Mitsuba Renderer](https://www.mitsuba-renderer.org/) --- scene format reference
- [TinyEXR](https://github.com/syoyo/tinyexr) --- HDR image I/O
- [NanoGUI](https://github.com/wjakob/nanogui) --- UI framework

## License

This project is created for educational purposes.