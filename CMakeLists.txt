cmake_minimum_required(VERSION 4.1)
project(LEAP)

set(CMAKE_CXX_STANDARD 20)

add_executable(LEAP main.cpp)

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

        # 2. Add the include headers (omp.h)
        target_include_directories(LEAP PRIVATE ${LIBOMP_PREFIX}/include)

        # 3. Add the library search path
        target_link_directories(LEAP PRIVATE ${LIBOMP_PREFIX}/lib)

        # 4. Add the specific flags Apple Clang needs
        # We must use -Xpreprocessor because Apple Clang doesn't natively recognize -fopenmp
        target_compile_options(LEAP PRIVATE -Xpreprocessor -fopenmp)

        # 5. Link the library (libomp.dylib)
        target_link_libraries(LEAP PRIVATE omp)

    else ()
        message(FATAL_ERROR "Could not find libomp. Please run: brew install libomp")
    endif ()

    # --- Standard Setup (Linux/Windows) ---
else ()
    find_package(OpenMP REQUIRED)
    target_link_libraries(LEAP PRIVATE OpenMP::OpenMP_CXX)
endif ()