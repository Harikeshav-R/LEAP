cmake_minimum_required(VERSION 4.1)
project(LEAP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------------------
# 1. Dependency Setup
# ------------------------------------------------------------------------------

# --- LibTorch (Only needed for tokenizer target) ---
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third-party/libtorch" ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

# --- Tokenizers ---
add_subdirectory(third-party/tokenizers)

# --- OpenMP Configuration (Required for both) ---
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
# 2. Define Executables
# ------------------------------------------------------------------------------

add_executable(tokenizer src/tokenizer/main.cpp)
add_executable(inference src/inference/main.cpp)

# ------------------------------------------------------------------------------
# 3. Link Dependencies
# ------------------------------------------------------------------------------

# --- Target: Tokenizer ---
# Needs: LibTorch + Tokenizers Lib + OpenMP
target_link_libraries(tokenizer PRIVATE "${TORCH_LIBRARIES}" tokenizers)

# --- Target: Inference ---
# Needs: OpenMP Only (No Torch)
# If inference needs the tokenizer library, add 'tokenizers' here too.
# target_link_libraries(inference PRIVATE tokenizers)

# ------------------------------------------------------------------------------
# 4. OpenMP Linking (Shared Logic)
# ------------------------------------------------------------------------------

set(OMP_TARGETS tokenizer inference)

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