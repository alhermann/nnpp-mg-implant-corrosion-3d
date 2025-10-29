cmake_minimum_required(VERSION 3.0)
project(main)

include_directories(libraries/libkdtree++)
include_directories(src/helpers)
include_directories(/gpfs/home/hermann/local/include/eigen3)

set(SOURCES src/main.cpp)

# General C++11 requirement
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Flags for Release build
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -msse2 -DNDEBUG -fopenmp")

# Flags for Debug build
set(CMAKE_CXX_FLAGS_DEBUG "-g -msse2 -fopenmp")

add_executable(${PROJECT_NAME} ${SOURCES})