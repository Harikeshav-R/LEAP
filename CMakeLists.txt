cmake_minimum_required(VERSION 3.28)
project(LEAP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# ------------------------------------------------------------------------------
# 0. Optimization Flags & Setup
# ------------------------------------------------------------------------------

# Check for LTO/IPO support (Link Time Optimization)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_output)
if (ipo_supported)
    message(STATUS "IPO / LTO supported. Enabling for Release builds.")
else ()
    message(STATUS "IPO / LTO not supported: ${ipo_output}")
endif ()

# Apply aggressive optimizations for Release builds
if (NOT MSVC)
    # -march=native: Enable all instructions supported by the compiling machine (AVX2, NEON, etc.)
    # -mtune=native: Tune for the compiling machine
    # -ffast-math: Allow aggressive floating point optimizations (breaks strict IEEE compliance, good for AI)
    # -funroll-loops: Aggressive loop unrolling
    # -fomit-frame-pointer: Free up a register
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -ffast-math -funroll-loops -fomit-frame-pointer")
else ()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /arch:AVX2 /fp:fast")
endif ()

# ------------------------------------------------------------------------------
# 1. Dependency Setup
# ------------------------------------------------------------------------------

# --- Json ---
include(FetchContent)
FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
)
FetchContent_MakeAvailable(json)

# --- SafeTensors ---
FetchContent_Declare(
        safetensors_cpp
        GIT_REPOSITORY https://github.com/syoyo/safetensors-cpp
        GIT_TAG main
)
FetchContent_MakeAvailable(safetensors_cpp)

# --- LibTorch ---
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third-party/libtorch" ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

if (CMAKE_CXX_FLAGS)
    string(REPLACE "-fno-exceptions" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif ()

# Also clean the specific Torch variable if it exists
if (TORCH_CXX_FLAGS)
    string(REPLACE "-fno-exceptions" "" TORCH_CXX_FLAGS "${TORCH_CXX_FLAGS}")
endif ()

if (CMAKE_CXX_FLAGS_RELEASE)
    string(REPLACE "-fno-exceptions" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endif ()
# ================

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")

# Force exceptions globally to be safe
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")

# --- Tokenizers ---
add_subdirectory("${CMAKE_SOURCE_DIR}/third-party/tokenizers")

# --- OpenMP Configuration ---
set(USE_CUSTOM_OMP OFF)
if (APPLE)
    find_program(BREW_PROG brew)
    if (BREW_PROG)
        execute_process(COMMAND ${BREW_PROG} --prefix libomp
                OUTPUT_VARIABLE LIBOMP_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE)
    else ()
        if (EXISTS "/opt/homebrew/opt/libomp")
            set(LIBOMP_PREFIX "/opt/homebrew/opt/libomp")
        elseif (EXISTS "/usr/local/opt/libomp")
            set(LIBOMP_PREFIX "/usr/local/opt/libomp")
        endif ()
    endif ()

    if (LIBOMP_PREFIX)
        message(STATUS "Found Homebrew OpenMP at: ${LIBOMP_PREFIX}")
        set(USE_CUSTOM_OMP ON)
    else ()
        message(FATAL_ERROR "Could not find libomp. Please run: brew install libomp")
    endif ()
else ()
    find_package(OpenMP REQUIRED)
endif ()

# ------------------------------------------------------------------------------
# 2. Define Executables & Libraries
# ------------------------------------------------------------------------------

add_executable(export
        src/export/Export.cpp
        src/export/Export.h
        src/export/Loader.cpp
        src/export/Loader.h
        src/export/main.cpp
        src/export/Utils.cpp
        src/export/Utils.h
)

target_include_directories(export PRIVATE third-party/tokenizers/third-party/json/single_include ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(inference
        src/inference/Config.h
        src/inference/FloatTransformer.cpp
        src/inference/FloatTransformer.h
        src/inference/main.cpp
        src/inference/QuantizedTransformer.cpp
        src/inference/QuantizedTransformer.h
        src/inference/Sampler.cpp
        src/inference/Sampler.h
        src/inference/TcpTransport.cpp
        src/inference/TcpTransport.h
        src/inference/Tokenizer.cpp
        src/inference/Tokenizer.h
        src/inference/Transformer.h
        src/inference/TransformerFactory.cpp
        src/inference/Transport.h
        src/inference/Utils.h
)

add_library(model
        src/model/Attention.cpp
        src/model/Attention.h
        src/model/FeedForward.cpp
        src/model/FeedForward.h
        src/model/ModelArgs.h
        src/model/RMSNorm.cpp
        src/model/RMSNorm.h
        src/model/TransformerBlock.cpp
        src/model/TransformerBlock.h
        src/model/Transformer.cpp
        src/model/Transformer.h
        src/model/Utils.cpp
        src/model/Utils.h
)

add_executable(tokenizer
        src/tokenizer/main.cpp
        src/tokenizer/Tokenizer.cpp
        src/tokenizer/Tokenizer.h
)

# ------------------------------------------------------------------------------
# 3. Link Dependencies
# ------------------------------------------------------------------------------

# --- Target: Export ---
target_link_libraries(export PRIVATE ${TORCH_LIBRARIES} model nlohmann_json::nlohmann_json safetensors_cpp)

# --- Target: Inference ---
# Note: -ffast-math is already enabled globally above, but explicit options for headers are fine
target_compile_options(inference PRIVATE $<$<CONFIG:Release>:-ffast-math>)

# --- Target: Model ---
target_include_directories(model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/model)
target_link_libraries(model PUBLIC ${TORCH_LIBRARIES})

# --- Target: Tokenizer ---
target_link_libraries(tokenizer PRIVATE ${TORCH_LIBRARIES} tokenizers)

target_compile_options(export PRIVATE -fexceptions)
target_compile_options(model PRIVATE -fexceptions)
target_compile_options(tokenizer PRIVATE -fexceptions)

# ------------------------------------------------------------------------------
# 4. OpenMP Linking (Shared Logic)
# ------------------------------------------------------------------------------

set(OMP_TARGETS export inference model tokenizer)

foreach (TARGET_NAME ${OMP_TARGETS})
    if (APPLE AND USE_CUSTOM_OMP)
        target_include_directories(${TARGET_NAME} PRIVATE ${LIBOMP_PREFIX}/include)
        target_link_directories(${TARGET_NAME} PRIVATE ${LIBOMP_PREFIX}/lib)
        target_compile_options(${TARGET_NAME} PRIVATE -Xpreprocessor -fopenmp)
        target_link_libraries(${TARGET_NAME} PRIVATE omp)
    else ()
        target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif ()

    # Enable LTO if supported
    if (ipo_supported)
        set_property(TARGET ${TARGET_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif ()
endforeach ()
