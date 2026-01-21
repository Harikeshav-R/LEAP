cmake_minimum_required(VERSION 4.1)
project(LEAP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -O3")

# ------------------------------------------------------------------------------
# 1. Dependency Setup
# ------------------------------------------------------------------------------

# --- LibTorch ---
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third-party/libtorch" ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

# --- Tokenizers ---
add_subdirectory(third-party/tokenizers)

# --- OpenMP Configuration (Required for all) ---
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
        src/export/main.cpp
        src/model/ModelArgs.h
)

target_include_directories(export PRIVATE third-party/tokenizers/third-party/json/single_include ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(inference
        src/inference/main.cpp
)

add_library(model
        src/model/ModelArgs.h
        src/model/RMSNorm.cpp
        src/model/RMSNorm.h
        src/model/Utils.cpp
        src/model/Utils.h
        src/model/Attention.cpp
        src/model/Attention.h
        src/model/FeedForward.cpp
        src/model/FeedForward.h
        src/model/TransformerBlock.cpp
        src/model/TransformerBlock.h
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
# Needs: LibTorch + OpenMP
target_link_libraries(export PRIVATE ${TORCH_LIBRARIES} model)

# --- Target: Inference ---
# Needs: OpenMP Only (No Torch)
# If inference needs the tokenizer library, add 'tokenizers' here too.
# target_link_libraries(inference PRIVATE tokenizers)
target_compile_options(inference PRIVATE $<$<CONFIG:Release>:-ffast-math>)

# --- Target: Model ---
# Needs: LibTorch
target_include_directories(model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/model)
target_link_libraries(model PUBLIC ${TORCH_LIBRARIES})

# --- Target: Tokenizer ---
# Needs: LibTorch + Tokenizers Lib + OpenMP
target_link_libraries(tokenizer PRIVATE ${TORCH_LIBRARIES} tokenizers)

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
endforeach ()