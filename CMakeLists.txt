cmake_minimum_required(VERSION 3.0)
project(main CXX)

# -----------------------------
# Options
# -----------------------------
option(USE_KDTREE "Use kd-tree backend for neighborhood construction" OFF)

# -----------------------------
# Language standard
# -----------------------------
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------
# Sources
# -----------------------------
set(SOURCES
    src/main.cpp
    # add other .cpp files here if you create them
)

# -----------------------------
# Target
# -----------------------------
add_executable(${PROJECT_NAME} ${SOURCES})

# -----------------------------
# Includes (prefer vendored headers)
#   Expect:
#     ./eigen/Eigen/Dense
#     ./src/helpers/*.h
#     ./libraries/libkdtree++/kdtree++/*
# -----------------------------
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src/helpers
    ${CMAKE_SOURCE_DIR}/eigen
    ${CMAKE_SOURCE_DIR}/libraries/libkdtree++
)

# Provide a definition for kd-tree switching if your adapter uses it
if(USE_KDTREE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_KDTREE=1)
endif()

# -----------------------------
# Build type flags
# (preserve your original intent: -O3/-g, -msse2, NDEBUG in Release)
# -----------------------------
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Release>:-O3 -msse2 -DNDEBUG>
    $<$<CONFIG:Debug>:-g -msse2>
)

# -----------------------------
# OpenMP (portable handling)
# -----------------------------
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found. Building without OpenMP support.")
endif()

# -----------------------------
# Output dir convenience
# -----------------------------
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# -----------------------------
# Helpful messages
# -----------------------------
message(STATUS "Using vendored Eigen from: ${CMAKE_SOURCE_DIR}/eigen")
message(STATUS "KD-Tree backend: ${USE_KDTREE}")
