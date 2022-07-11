cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(betterphasefun VERSION 0.1 DESCRIPTION "Fruzsina's smarter ODE solver" LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)

# Set preprocessor macros
add_compile_definitions(BPHF_DIR="${PROJECT_SOURCE_DIR}/")

# Must use GNUInstallDirs to install libraries into correct locations on all platforms.
include(GNUInstallDirs)
# FetchContent helps with Pybind11 if it cannot be found
include(FetchContent)

# Set some compiler flags for non-release-type builds (okay for a small library)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(pybind11 2.6.0 QUIET)
if(NOT PYBIND_FOUND)
    find_package(PythonInterp 3 QUIET)
    if(PYTHONINTERP_FOUND)
        set(INTERNAL_PYBIND11_DIR "${PROJECT_SOURCE_DIR}/ext/pybind11")
        FetchContent_Declare(pybind11
                             GIT_REPOSITORY https://github.com/pybind/pybind11.git
                             GIT_TAG v2.9.0
                             GIT_PROGRESS ON
                             GIT_SHALLOW ON
                             GIT_SUBMODULES ""
                             GIT_CONFIG advice.detachedHead=False
                             SOURCE_DIR ${INTERNAL_PYBIND11_DIR})
        if(EXISTS ${INTERNAL_PYBIND11_DIR})
            message("-- Python 3 and pybind11 found!")
            set(PYBIND11_INSTALLED 1)
        else()
            message("-- Python 3 found but Pyind11 was not. Will now download pybind11 header files...")
            FetchContent_Populate(pybind11)
        endif()
        add_subdirectory(${INTERNAL_PYBIND11_DIR})
    else()
       message(FATAL_ERROR "-- Could not find Python 3. Please provide a hint to CMAKE.")
    endif()
endif()

add_subdirectory(src)
