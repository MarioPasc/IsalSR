cmake_minimum_required(VERSION 3.18)
project(isalsr_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------
# ISA target selection.
#
# Default: x86-64-v3 (AVX2 + FMA + POPCNT).  This is the lowest common
# denominator across the four Picasso node classes: 178 of 333 nodes lack
# AVX-512, so -march=native would produce a non-portable binary.  One flag
# set → one build hash across the entire 3,000-run campaign.
#
# Local override: -DISALSR_NATIVE_MARCH=ON swaps x86-64-v3 for native.
# Use this on the workstation for profiling; do NOT submit to SLURM.
# -----------------------------------------------------------------------
option(ISALSR_NATIVE_MARCH "Replace -march=x86-64-v3 with -march=native" OFF)

# -----------------------------------------------------------------------
# Sanitizer support (development only — never combine with Release/LTO)
# -----------------------------------------------------------------------
option(ISALSR_ENABLE_SANITIZERS "Enable Address + Undefined sanitizers" OFF)

# -----------------------------------------------------------------------
# Python + nanobind
# NOTE: find_package must come before nanobind_add_module.
# -----------------------------------------------------------------------
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# -----------------------------------------------------------------------
# Extension module: isalsr.core._native
#
# Sources live in src/isalsr/core/native/, alongside the Python package.
# The directory is named `native`, NOT `_native`: the built extension is
# `isalsr.core._native`, and a sibling directory of that exact name could be
# picked up as an implicit namespace package.  CPython's FileFinder tries
# extension loaders before the namespace fallback so the .so would still
# win, but the margin is one import-machinery detail and the names are free
# to differ — so they do.
#
# The C++ sources are excluded from the wheel via tool.scikit-build
# wheel.exclude in pyproject.toml; only the compiled .so is shipped.
# -----------------------------------------------------------------------
nanobind_add_module(_native
    NB_STATIC
    src/isalsr/core/native/src/bindings.cpp
    src/isalsr/core/native/src/probe.cpp
    src/isalsr/core/native/src/cdll.cpp
    src/isalsr/core/native/src/labeled_dag.cpp
    src/isalsr/core/native/src/node_types.cpp
    src/isalsr/core/native/src/string_to_dag.cpp
    src/isalsr/core/native/src/wl.cpp
    src/isalsr/core/native/src/canonical.cpp
)

target_include_directories(_native PRIVATE src/isalsr/core/native/include)

# -----------------------------------------------------------------------
# Per-target compile options — applied only to _native, NOT to
# nanobind-static or any other CMake target.  Using global
# add_compile_options() would propagate our flags into nanobind's own
# static library and trigger a GCC LTO/ODR redefinition bug in nb_types.h.
# -----------------------------------------------------------------------
# Warnings (always)
target_compile_options(_native PRIVATE -Wall -Wextra -Wpedantic)

if(ISALSR_ENABLE_SANITIZERS)
    target_compile_options(_native PRIVATE
        -fsanitize=address,undefined -fno-omit-frame-pointer -g -O1)
    target_link_options(_native PRIVATE
        -fsanitize=address,undefined)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(ISALSR_NATIVE_MARCH)
        target_compile_options(_native PRIVATE
            -O3 -march=native -DNDEBUG -fno-plt -funroll-loops)
    else()
        target_compile_options(_native PRIVATE
            -O3 -march=x86-64-v3 -DNDEBUG -fno-plt -funroll-loops)
    endif()

    # LTO / IPO: applied to _native target only (not to nanobind-static).
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _IPO_OK OUTPUT _IPO_ERR)
    if(_IPO_OK)
        set_property(TARGET _native PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()

# -----------------------------------------------------------------------
# Static libstdc++ / libgcc on Linux.
#
# Picasso's login-node default is gcc 7.5.0 (too old for C++17).  By
# linking statically we remove the "module load gcc/…" requirement from
# every SLURM task at run time.  The build itself still needs a C++17
# compiler (e.g. `module load gcc/13.2.0` on Picasso at build time).
# -----------------------------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(_native PRIVATE -static-libstdc++ -static-libgcc)
endif()

# -----------------------------------------------------------------------
# Install inside the isalsr.core package so `import isalsr.core._native`
# resolves correctly in both editable and regular installs.
# -----------------------------------------------------------------------
install(TARGETS _native LIBRARY DESTINATION isalsr/core)
