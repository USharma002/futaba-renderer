cmake_minimum_required(VERSION 3.18)

# Enable BOTH C++ and CUDA languages
project(app LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Find the CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Disable NanoGUI python bindings and examples
set(NANOGUI_BUILD_PYTHON OFF CACHE BOOL "Disable Python bindings" FORCE)
set(NANOGUI_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples" FORCE)
set(NANOGUI_BUILD_SHARED OFF CACHE BOOL "Build static library" FORCE)

add_subdirectory(ext/nanogui)

add_executable(app 
    main.cpp 
    src/gui/renderer_host.cpp 
    src/gui/gui.cpp
    src/films/bitmap.cpp
    src/parser/scene_loader.cpp 
    ext/pugixml/src/pugixml.cpp
    src/accel/optix_bvh.cu
)

find_path(OPTIX_INCLUDE_DIR optix.h
    HINTS
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0/include"
)
if(OPTIX_INCLUDE_DIR)
    target_include_directories(app PRIVATE ${OPTIX_INCLUDE_DIR})
endif()

# Force GLAD macro for Windows OpenGL
target_compile_definitions(app PRIVATE NANOGUI_GLAD)

set(APP_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/accel
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/films
    ${CMAKE_CURRENT_SOURCE_DIR}/src/frame
    ${CMAKE_CURRENT_SOURCE_DIR}/src/integrator
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sensors
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gui
    ${CMAKE_CURRENT_SOURCE_DIR}/src/scene
    ${CMAKE_CURRENT_SOURCE_DIR}/src/parser
    ${CMAKE_CURRENT_SOURCE_DIR}/src/shapes
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sampler
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bsdfs
    ${CMAKE_CURRENT_SOURCE_DIR}/src/emitters
    ${CMAKE_CURRENT_SOURCE_DIR}/src/materials
    ${CMAKE_CURRENT_SOURCE_DIR}/src/medium
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/nanogui/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/pugixml/src
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/tinyexr
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/nanogui/ext/glad/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/nanogui/ext/nanovg/src
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/nanogui/ext/glfw/include
)

target_include_directories(app PRIVATE ${APP_INCLUDES})

# Build renderer_device.cu as PTX
set(PTX_OUTPUT_DIR "${CMAKE_BINARY_DIR}/ptx")
set(PTX_FILE "${PTX_OUTPUT_DIR}/renderer_device.ptx")

# Convert includes to -I flags
set(NVCC_INCLUDES "")
foreach(INC IN LISTS APP_INCLUDES)
    list(APPEND NVCC_INCLUDES "-I${INC}")
endforeach()
if(OPTIX_INCLUDE_DIR)
    list(APPEND NVCC_INCLUDES "-I${OPTIX_INCLUDE_DIR}")
endif()

add_custom_command(
    OUTPUT "${PTX_FILE}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PTX_OUTPUT_DIR}"
    COMMAND ${CMAKE_CUDA_COMPILER} --ptx -std=c++17 -O3 --use_fast_math
            -DFUTABA_OPTIX_DEVICE_PROGRAMS
            ${NVCC_INCLUDES}
            "${CMAKE_CURRENT_SOURCE_DIR}/src/gui/renderer_device.cu"
            -o "${PTX_FILE}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/gui/renderer_device.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/gui/launch_params.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/integrator/heatmap.cuh"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/accel/bvh.cuh"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/scene/scene.cuh"
)
add_custom_target(compile_ptx DEPENDS "${PTX_FILE}")
add_dependencies(app compile_ptx)

string(REPLACE "\\" "/" PTX_FILE_FORWARD_SLASHES "${PTX_FILE}")
target_compile_definitions(app PRIVATE "PTX_FILE_PATH=\"${PTX_FILE_FORWARD_SLASHES}\"")

string(REPLACE "\\" "/" FUTABA_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(app PRIVATE FUTABA_SOURCE_ROOT="${FUTABA_SOURCE_ROOT}")

# 4. Link NanoGUI AND the CUDA runtime
target_link_libraries(app PRIVATE 
    nanogui 
    CUDA::cudart
)