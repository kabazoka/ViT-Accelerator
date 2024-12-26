cmake_minimum_required(VERSION 3.12)
project(vit
        DESCRIPTION "Inference Vision Transformer (ViT) in plain C/C++ with ggml"
        HOMEPAGE_URL "https://github.com/staghado/vit.cpp")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set the compiler flags, modify here if needed!
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# You might enable this at your own risk (it can affect numerical accuracy):
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")

# ------------------------------------------------------------------------------
# GGML dependency
# ------------------------------------------------------------------------------
add_subdirectory(ggml)

# ------------------------------------------------------------------------------
# Option to enable Xilinx FPGA offload
# ------------------------------------------------------------------------------
option(USE_FPGA "Enable FPGA acceleration with XRT" OFF)

# If USE_FPGA is ON, we link to XRT libraries, etc. Otherwise, we skip them.
set(VIT_EXTRA_LIBS "")

# Adjust these paths to match your local XRT installation
set(XRT_INCLUDE_DIR "/opt/xilinx/xrt/include")
set(GGML_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/ggml/include")
set(XRT_LIB_DIR     "/opt/xilinx/xrt/lib")

include_directories(${XRT_INCLUDE_DIR} ${GGML_INCLUDE_DIR})
link_directories(${XRT_LIB_DIR})

# Typically needed libraries for XRT-based flow
list(APPEND VIT_EXTRA_LIBS xrt_coreutil OpenCL pthread)
message(STATUS "Linking XRT + OpenCL libraries.")

# ------------------------------------------------------------------------------
# Main 'vit' executable
# ------------------------------------------------------------------------------
add_executable(${PROJECT_NAME}
    main.cpp
    vit.cpp
)

# Link standard libraries, plus extra if needed
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        ggml
        ${VIT_EXTRA_LIBS}
)

target_include_directories(${PROJECT_NAME} PUBLIC .)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

# ------------------------------------------------------------------------------
# quantize executable
# ------------------------------------------------------------------------------
option(BUILD_QUANTIZE "Build the quantization executable" ON)
if (BUILD_QUANTIZE)
    add_executable(quantize quantize.cpp)
    target_link_libraries(quantize PUBLIC ggml)
    target_include_directories(quantize PUBLIC .)
    target_compile_features(quantize PUBLIC cxx_std_11)
endif()

# ------------------------------------------------------------------------------
# benchmark executable (optional)
# ------------------------------------------------------------------------------
option(BUILD_BENCHMARK "Build the benchmark executable" OFF)
if(BUILD_BENCHMARK)
    # requires nlohmann_json
    find_package(nlohmann_json 3.2.0 REQUIRED)

    add_executable(benchmark tests/benchmark.cpp vit.cpp)
    target_link_libraries(benchmark PRIVATE nlohmann_json::nlohmann_json ggml)
    target_include_directories(benchmark PUBLIC .)
    target_compile_features(benchmark PUBLIC cxx_std_17)
endif()
