# Install script for directory: /gpfs/home/hermann/eigen/Eigen

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
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/Eigen" TYPE FILE FILES
    "/gpfs/home/hermann/eigen/Eigen/QtAlignedMalloc"
    "/gpfs/home/hermann/eigen/Eigen/CholmodSupport"
    "/gpfs/home/hermann/eigen/Eigen/MetisSupport"
    "/gpfs/home/hermann/eigen/Eigen/SuperLUSupport"
    "/gpfs/home/hermann/eigen/Eigen/SPQRSupport"
    "/gpfs/home/hermann/eigen/Eigen/Eigen"
    "/gpfs/home/hermann/eigen/Eigen/Core"
    "/gpfs/home/hermann/eigen/Eigen/StdDeque"
    "/gpfs/home/hermann/eigen/Eigen/SparseQR"
    "/gpfs/home/hermann/eigen/Eigen/SparseCore"
    "/gpfs/home/hermann/eigen/Eigen/Eigenvalues"
    "/gpfs/home/hermann/eigen/Eigen/Geometry"
    "/gpfs/home/hermann/eigen/Eigen/SVD"
    "/gpfs/home/hermann/eigen/Eigen/OrderingMethods"
    "/gpfs/home/hermann/eigen/Eigen/UmfPackSupport"
    "/gpfs/home/hermann/eigen/Eigen/Cholesky"
    "/gpfs/home/hermann/eigen/Eigen/SparseCholesky"
    "/gpfs/home/hermann/eigen/Eigen/StdList"
    "/gpfs/home/hermann/eigen/Eigen/QR"
    "/gpfs/home/hermann/eigen/Eigen/PaStiXSupport"
    "/gpfs/home/hermann/eigen/Eigen/Sparse"
    "/gpfs/home/hermann/eigen/Eigen/Householder"
    "/gpfs/home/hermann/eigen/Eigen/PardisoSupport"
    "/gpfs/home/hermann/eigen/Eigen/SparseLU"
    "/gpfs/home/hermann/eigen/Eigen/StdVector"
    "/gpfs/home/hermann/eigen/Eigen/Jacobi"
    "/gpfs/home/hermann/eigen/Eigen/Dense"
    "/gpfs/home/hermann/eigen/Eigen/IterativeLinearSolvers"
    "/gpfs/home/hermann/eigen/Eigen/LU"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Devel")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Devel")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/Eigen" TYPE DIRECTORY FILES "/gpfs/home/hermann/eigen/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Devel")

