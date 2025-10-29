# Install script for directory: /gpfs/home/hermann/eigen/unsupported/Eigen

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/gpfs/home/hermann/local")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "Release")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

# Install shared libraries without execute permission?
IF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  SET(CMAKE_INSTALL_SO_NO_EXE "0")
ENDIF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Devel")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "/gpfs/home/hermann/eigen/unsupported/Eigen/AdolcForward"
    "/gpfs/home/hermann/eigen/unsupported/Eigen/AlignedVector3"
    "/gpfs/home/hermann/eigen/unsupported/Eigen/ArpackSupport"
    "/gpfs/home/hermann/eigen/unsupported/Eigen/AutoDiff"
    "/gpfs/home/hermann/eigen/unsupported/Eigen/BVH"
    "/gpfs/home/hermann/eigen/unsupported/Eigen/EulerAngles"
    "/gpfs/home/hermann/eigen/unsupported/Eigen/FFT"
    "/gpfs/home/hermann/eigen/unsupported/Eigen/IterativeSolvers"
    "/gpfs/home/hermann/eigen/unsupported/Eigen/KroneckerProduct"
    "/gpfs/home/hermann/eigen/unsupported/Eigen/LevenbergMarquardt"
    "/gpfs/home/hermann/eigen/unsupported/Eigen/MatrixFunctions"
    "/gpfs/home/hermann/eigen/unsupported/Eigen/MoreVectorization"
    "/gpfs/home/hermann/eigen/unsupported/Eigen/MPRealSupport"
    "/gpfs/home/hermann/eigen/unsupported/Eigen/NonLinearOptimization"
    "/gpfs/home/hermann/eigen/unsupported/Eigen/NumericalDiff"
    "/gpfs/home/hermann/eigen/unsupported/Eigen/OpenGLSupport"
    "/gpfs/home/hermann/eigen/unsupported/Eigen/Polynomials"
    "/gpfs/home/hermann/eigen/unsupported/Eigen/Skyline"
    "/gpfs/home/hermann/eigen/unsupported/Eigen/SparseExtra"
    "/gpfs/home/hermann/eigen/unsupported/Eigen/SpecialFunctions"
    "/gpfs/home/hermann/eigen/unsupported/Eigen/Splines"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Devel")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Devel")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "/gpfs/home/hermann/eigen/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Devel")

IF(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  INCLUDE("/gpfs/home/hermann/eigen/build/unsupported/Eigen/CXX11/cmake_install.cmake")

ENDIF(NOT CMAKE_INSTALL_LOCAL_ONLY)

