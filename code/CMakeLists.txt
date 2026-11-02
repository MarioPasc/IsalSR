cmake_minimum_required(VERSION 3.18)
# LANGUAGES NONE: the C++ engine is an accelerator, not a requirement.
# isalsr.core.backends falls back to the pure-Python engine when
# isalsr.core._native is absent, and the two produce identical strings, so a
# machine without a usable toolchain still gets a working install instead of
# a failed `pip install`.  Every `return()` below installs the Python package
# alone.
project(isalsr_native LANGUAGES NONE)

option(ISALSR_BUILD_NATIVE "Build the C++ engine isalsr.core._native" ON)
if(NOT ISALSR_BUILD_NATIVE)
    message(STATUS "ISALSR_BUILD_NATIVE=OFF: installing the pure-Python engine only")
    return()
endif()

include(CheckLanguage)
check_language(CXX)
if(NOT CMAKE_CXX_COMPILER)
    message(WARNING "No C++ compiler found: installing the pure-Python engine only. "
                    "Install a C++17 compiler (GCC >= 11 or Clang) and reinstall for the C++ engine.")
    return()
endif()
enable_language(CXX)

# The engine is built and tested with GCC and Clang.  MSVC is untested, so a
# Windows source build installs the pure-Python engine rather than risk a
# failed install.
if(MSVC)
    message(WARNING "MSVC is not a supported compiler for the C++ engine: "
                    "installing the pure-Python engine only.")
    return()
endif()

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
#
# ISALSR_MARCH sets the x86-64 level.  Keep the default to reproduce the
# paper's build hash; the PyPI wheels use x86-64-v2 so that they import on
# CPUs without AVX2 instead of dying with SIGILL.  -march is an x86 flag:
# other architectures (aarch64, Apple silicon) build for the compiler's
# default target.
# -----------------------------------------------------------------------
option(ISALSR_NATIVE_MARCH "Replace -march=x86-64-v3 with -march=native" OFF)
set(ISALSR_MARCH "x86-64-v3" CACHE STRING "Value of -march= on x86-64")

# -----------------------------------------------------------------------
# Sanitizer support (development only — never combine with Release/LTO)
# -----------------------------------------------------------------------
option(ISALSR_ENABLE_SANITIZERS "Enable Address + Undefined sanitizers" OFF)

# -----------------------------------------------------------------------
# Python + nanobind
# NOTE: find_package must come before nanobind_add_module.
# -----------------------------------------------------------------------
find_package(Python 3.11 COMPONENTS Interpreter Development.Module)
if(NOT Python_Development.Module_FOUND)
    message(WARNING "Python development headers not found: installing the pure-Python "
                    "engine only (on Debian/Ubuntu: apt install python3-dev).")
    return()
endif()
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
        set(_isalsr_march -march=native)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        set(_isalsr_march -march=${ISALSR_MARCH})
    else()
        set(_isalsr_march "")
    endif()
    target_compile_options(_native PRIVATE
        -O3 ${_isalsr_march} -DNDEBUG -fno-plt -funroll-loops)

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
