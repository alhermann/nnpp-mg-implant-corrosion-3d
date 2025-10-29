// Include, system

#define _USE_MATH_DEFINES
#include <cmath>

#include <omp.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/QR>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include "unsupported/Eigen/src/IterativeSolvers/GMRES.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <map>

#include "kd_tree_interface.h"
#include "space.h"
#include "utils.h"
#include "grid.h"

/// Macros

#define DPN 3
#define MAX_HORIZON_LENGTH 1024
#define EPS 10E-6
#define MAX_ITER 200

// Simulation flags
int sub_step_no = 8;
int BunnyConditions = 1;

// Domain parameters

int maxHorizonLength;
int maxSurfaceHorizonLength;
double meanHorizonLength;
double meanHorizonBunnyLength;

int solidCount = 0;

std::size_t s0 = 1;

double activeKappa = 0;

double energyValue;

double res1Old;
double res2Old;
double res3Old;
double res4Old;
double res5Old;
double res6Old;
double res7Old;
double res8Old;
double res9Old;
double res10Old;
double res11Old;

double res1;
double res2;
double res3;
double res4;
double res5;
double res6;
double res7;
double res8;
double res9;
double res10;
double res11;

double resMax;

Eigen::VectorXd controlData1;
Eigen::VectorXd controlData2;

Eigen::VectorXd controlDataC1;
Eigen::VectorXd controlDataC2;
Eigen::VectorXd controlDataC3;
Eigen::VectorXd controlDataC4;
Eigen::VectorXd controlDataC5;
Eigen::VectorXd controlDataC6;
Eigen::VectorXd controlDataC7;
Eigen::VectorXd controlDataC8;
Eigen::VectorXd controlDataC9;
Eigen::VectorXd controlDataC10;
Eigen::VectorXd controlDataC11;

Eigen::VectorXd F1;
Eigen::VectorXd F2;
Eigen::VectorXd F3;
Eigen::VectorXd F4;
Eigen::VectorXd F5;
Eigen::VectorXd F6;
Eigen::VectorXd F7;
Eigen::VectorXd F8;
Eigen::VectorXd F9;
Eigen::VectorXd F10;
Eigen::VectorXd F11;

Eigen::MatrixXd nodes;

Eigen::VectorXi boundaryConditionTypes;
Eigen::VectorXd boundaryConditionValues1;
Eigen::VectorXd boundaryConditionValues2;
Eigen::VectorXd boundaryConditionValues3;
Eigen::VectorXd boundaryConditionValues4;
Eigen::VectorXd boundaryConditionValues5;
Eigen::VectorXd boundaryConditionValues6;
Eigen::VectorXd boundaryConditionValues7;
Eigen::VectorXd boundaryConditionValues8;
Eigen::VectorXd boundaryConditionValues9;
Eigen::VectorXd boundaryConditionValues10;
Eigen::VectorXd boundaryConditionValues11;

Eigen::VectorXd initialConditionValues1;
Eigen::VectorXd initialConditionValues2;
Eigen::VectorXd initialConditionValues3;
Eigen::VectorXd initialConditionValues4;
Eigen::VectorXd initialConditionValues5;
Eigen::VectorXd initialConditionValues6;
Eigen::VectorXd initialConditionValues7;
Eigen::VectorXd initialConditionValues8;
Eigen::VectorXd initialConditionValues9;
Eigen::VectorXd initialConditionValues10;
Eigen::VectorXd initialConditionValues11;

Eigen::VectorXi horizonsLengths;
Eigen::MatrixXi horizons;

Eigen::VectorXi horizonsBunnyLengths;
Eigen::MatrixXi horizonsBunny;

Eigen::MatrixXd bunnyNodes;

typedef Eigen::SparseMatrix<double> SpMat;

// Define random uniform distribution
std::random_device
    rd;  // Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
std::uniform_real_distribution<> dist(0, 1);

int main(int argc, char* argv[]) {
  std::cout << "    _/    _/  _/_/_/_/  _/_/_/    _/_/_/_/    _/_/    _/      "
               "_/       "
            << "\n"
            << "   _/    _/  _/        _/    _/  _/        _/    _/  _/_/    "
               "_/        "
            << "\n"
            << "  _/_/_/_/  _/_/_/    _/_/_/    _/_/_/    _/    _/  _/  _/  _/ "
               "        "
            << "\n"
            << " _/    _/  _/        _/    _/  _/        _/    _/  _/    _/_/  "
               "        "
            << "\n"
            << "_/    _/  _/_/_/_/  _/    _/  _/_/_/_/    _/_/    _/      _/   "
               "        "
            << std::endl;

  // Check command line parameters

  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <prefix> <jobname> <startfileNo>"
              << std::endl;

    return 0;
  }

  std::string Prefix = argv[1];
  std::string JobName = argv[2];
  std::size_t StartFileNo = std::stoi(argv[3]);

#ifdef _OPENMP

  // omp_set_dynamic(0);
  // omp_set_num_threads(10);

  Eigen::initParallel();

#pragma omp parallel
  {
    if (omp_get_thread_num() == 0) {
      std::cout << "OpenMP enabled, no. of cores: " << omp_get_num_threads()
                << std::endl;
    }
  }

#else  // _OPENMP

  std::cout << "OpenMP disabled." << std::endl;

#endif  // _OPENMP

  // Eigen Version
  std::cout << "Eigen library version: " << EIGEN_WORLD_VERSION << "."
            << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << std::endl;

  // Reading data

  utils::read_vector_from_file(JobName + ".data1", controlData1, 4);
  utils::read_vector_from_file(JobName + ".data2", controlData2, 51);

  utils::read_vector_from_file(JobName + ".datac1", controlDataC1, 7);
  utils::read_vector_from_file(JobName + ".datac2", controlDataC2, 7);
  utils::read_vector_from_file(JobName + ".datac3", controlDataC3, 7);
  utils::read_vector_from_file(JobName + ".datac4", controlDataC4, 7);
  utils::read_vector_from_file(JobName + ".datac5", controlDataC5, 7);
  utils::read_vector_from_file(JobName + ".datac6", controlDataC6, 7);
  utils::read_vector_from_file(JobName + ".datac7", controlDataC7, 7);
  utils::read_vector_from_file(JobName + ".datac8", controlDataC8, 7);
  utils::read_vector_from_file(JobName + ".datac9", controlDataC9, 7);
  utils::read_vector_from_file(JobName + ".datac10", controlDataC10, 7);
  utils::read_vector_from_file(JobName + ".datac11", controlDataC11, 4);

  // Parameter class holder
  utils::parameter params;

  params.set_int(controlData1);
  params.set_double(controlData2, controlDataC1, controlDataC2, controlDataC3,
                    controlDataC4, controlDataC5, controlDataC6, controlDataC7,
                    controlDataC8, controlDataC9, controlDataC10, controlDataC11);

  std::cout << "Reading data... ";
  utils::read_matrix_from_file(JobName + ".nodes", nodes, params.nodesNo, DPN);
  
  // Read bunny nodes
  if (BunnyConditions == 1) {
  	utils::read_matrix_from_file(JobName + ".bunnyNodes", bunnyNodes,
                               	params.bunnyNodesNo, DPN);
  }

  double energyBound = nodes.maxCoeff();

  utils::read_vector_from_file(JobName + ".bctypes", boundaryConditionTypes,
                               params.nodesNo);
  utils::read_vector_from_file(JobName + ".bcvalues1", boundaryConditionValues1,
                               params.nodesNo);
  utils::read_vector_from_file(JobName + ".bcvalues2", boundaryConditionValues2,
                               params.nodesNo);
  utils::read_vector_from_file(JobName + ".bcvalues3", boundaryConditionValues3,
                               params.nodesNo);
  utils::read_vector_from_file(JobName + ".bcvalues4", boundaryConditionValues4,
                               params.nodesNo);
  utils::read_vector_from_file(JobName + ".bcvalues5", boundaryConditionValues5,
                               params.nodesNo);
  utils::read_vector_from_file(JobName + ".bcvalues6", boundaryConditionValues6,
                               params.nodesNo);
  utils::read_vector_from_file(JobName + ".bcvalues7", boundaryConditionValues7,
                               params.nodesNo);
  utils::read_vector_from_file(JobName + ".bcvalues8", boundaryConditionValues8,
                               params.nodesNo);
  utils::read_vector_from_file(JobName + ".bcvalues9", boundaryConditionValues9,
                               params.nodesNo);
  utils::read_vector_from_file(JobName + ".bcvalues10", boundaryConditionValues10,
                               params.nodesNo);
  utils::read_vector_from_file(JobName + ".bcvalues11", boundaryConditionValues11,
                               params.nodesNo);
  utils::read_vector_from_file(JobName + ".icvalues1", initialConditionValues1,
                               params.nodesNo);
  utils::read_vector_from_file(JobName + ".icvalues2", initialConditionValues2,
                               params.nodesNo);
  utils::read_vector_from_file(JobName + ".icvalues3", initialConditionValues3,
                               params.nodesNo);
  utils::read_vector_from_file(JobName + ".icvalues4", initialConditionValues4,
                               params.nodesNo);
  utils::read_vector_from_file(JobName + ".icvalues5", initialConditionValues5,
                               params.nodesNo);
  utils::read_vector_from_file(JobName + ".icvalues6", initialConditionValues6,
                               params.nodesNo);
  utils::read_vector_from_file(JobName + ".icvalues7", initialConditionValues7,
                               params.nodesNo);
  utils::read_vector_from_file(JobName + ".icvalues8", initialConditionValues8,
                               params.nodesNo);
  utils::read_vector_from_file(JobName + ".icvalues9", initialConditionValues9,
                               params.nodesNo);
  utils::read_vector_from_file(JobName + ".icvalues10", initialConditionValues10,
                               params.nodesNo);
  utils::read_vector_from_file(JobName + ".icvalues11", initialConditionValues11,
                               params.nodesNo);

  std::ofstream energyFile;
  std::ofstream volumeLossFile;

  std::cout << "finished." << std::endl;

  std::cout << "Domain boundary: " << energyBound << std::endl;

  utils::timer timer;

  // Building horizons

  timer.tic();

  build_clouds_hash(MAX_HORIZON_LENGTH, params.delta * (1.00 + 1e-5), params.dx * 10., params.dy * 10., params.dz * 10., 
	               nodes, horizons, horizonsLengths, maxHorizonLength);

  // build_clouds(0, MAX_HORIZON_LENGTH, params.delta * (1.00 + 1e-5), 0.00, nodes,
  //              horizons, horizonsLengths, maxHorizonLength, meanHorizonLength);
  //
  std::cout << "Building horizons took " << timer.toc()
            << " seconds. Max. horizon length: " << maxHorizonLength
  //           << " Mean horizon length: " << meanHorizonLength 
            << std::endl;
  std::cout << params.nodesNo << " nodes; horizon radius: " << params.delta << std::endl;
  
  if (BunnyConditions == 1) {
    build_clouds(0, MAX_HORIZON_LENGTH, 3. * 0.4 * params.dx * (1.00 + 1e-5), 0.00, nodes,
                 bunnyNodes, horizonsBunny, horizonsBunnyLengths,
                 maxHorizonLength, meanHorizonBunnyLength);

    std::cout << "Building bunnyNodes took " << timer.toc()
              << " seconds. Max. horizonsBunny length: " << maxHorizonLength
              << " \nMean horizonBunny length: " << meanHorizonBunnyLength
              << std::endl;
  }

  // Initialize the solution vector
  Eigen::VectorXd U1n0(params.nodesNo);
  Eigen::VectorXd U1n(params.nodesNo);
  Eigen::VectorXd U1n1(params.nodesNo);
  Eigen::VectorXd U2n0(params.nodesNo);
  Eigen::VectorXd U2n(params.nodesNo);
  Eigen::VectorXd U2n1(params.nodesNo);
  Eigen::VectorXd U3n0(params.nodesNo);
  Eigen::VectorXd U3n(params.nodesNo);
  Eigen::VectorXd U3n1(params.nodesNo);
  Eigen::VectorXd U4n0(params.nodesNo);
  Eigen::VectorXd U4n(params.nodesNo);
  Eigen::VectorXd U4n1(params.nodesNo);
  Eigen::VectorXd U5n0(params.nodesNo);
  Eigen::VectorXd U5n(params.nodesNo);
  Eigen::VectorXd U5n1(params.nodesNo);
  Eigen::VectorXd U6n0(params.nodesNo);
  Eigen::VectorXd U6n(params.nodesNo);
  Eigen::VectorXd U6n1(params.nodesNo);
  Eigen::VectorXd U7n0(params.nodesNo);
  Eigen::VectorXd U7n(params.nodesNo);
  Eigen::VectorXd U7n1(params.nodesNo);
  Eigen::VectorXd U8n0(params.nodesNo);
  Eigen::VectorXd U8n(params.nodesNo);
  Eigen::VectorXd U8n1(params.nodesNo);
  Eigen::VectorXd U9n0(params.nodesNo);
  Eigen::VectorXd U9n(params.nodesNo);
  Eigen::VectorXd U9n1(params.nodesNo);
  Eigen::VectorXd U10n0(params.nodesNo);
  Eigen::VectorXd U10n(params.nodesNo);
  Eigen::VectorXd U10n1(params.nodesNo);
  Eigen::VectorXd U11n0(params.nodesNo);
  Eigen::VectorXd U11n(params.nodesNo);
  Eigen::VectorXd U11n1(params.nodesNo);

  Eigen::VectorXi nodePhase(params.nodesNo);

  Eigen::VectorXd F1(params.nodesNo);
  Eigen::VectorXd dU1(params.nodesNo);
  Eigen::VectorXd F2(params.nodesNo);
  Eigen::VectorXd dU2(params.nodesNo);
  Eigen::VectorXd F3(params.nodesNo);
  Eigen::VectorXd dU3(params.nodesNo);
  Eigen::VectorXd F4(params.nodesNo);
  Eigen::VectorXd dU4(params.nodesNo);
  Eigen::VectorXd F5(params.nodesNo);
  Eigen::VectorXd dU5(params.nodesNo);
  Eigen::VectorXd F6(params.nodesNo);
  Eigen::VectorXd dU6(params.nodesNo);
  Eigen::VectorXd F7(params.nodesNo);
  Eigen::VectorXd dU7(params.nodesNo);
  Eigen::VectorXd F8(params.nodesNo);
  Eigen::VectorXd dU8(params.nodesNo);
  Eigen::VectorXd F9(params.nodesNo);
  Eigen::VectorXd dU9(params.nodesNo);
  Eigen::VectorXd F10(params.nodesNo);
  Eigen::VectorXd dU10(params.nodesNo);
  Eigen::VectorXd F11(params.nodesNo);
  Eigen::VectorXd dU11(params.nodesNo);

  // Initialize
  U1n0.setZero();
  U2n0.setZero();
  U3n0.setZero();
  U4n0.setZero();
  U5n0.setZero();
  U6n0.setZero();
  U7n0.setZero();
  U8n0.setZero();
  U9n0.setZero();
  U10n0.setZero();
  U11n0.setZero();

  if (BunnyConditions == 0) {
	  U1n = initialConditionValues1;
	  U2n = initialConditionValues2;
	  U3n = initialConditionValues3;
	  U4n = initialConditionValues4;
	  U5n = initialConditionValues5;
    U6n = initialConditionValues6;
	  U7n = initialConditionValues7;
	  U8n = initialConditionValues8;
	  U9n = initialConditionValues9;
	  U10n = initialConditionValues10;
    U11n = initialConditionValues11;
  
#pragma omp parallel for reduction(+ : solidCount)
	  for (int i = 0; i < params.nodesNo; i++) {
		  if (U1n[i] > 0.0 + EPS) {
			  nodePhase[i] = 1;
			  solidCount++;
		  } else {
			  nodePhase[i] = 0;
		  }
	  }
  } else if (BunnyConditions == 1) {
#pragma omp parallel for reduction(+ : solidCount)
      for (int i = 0; i < params.nodesNo; i++) {
        if (horizonsBunnyLengths[i] == 0) {
          // Liquid
          U1n[i] = 0.0;                                         // Mg2+
          nodePhase[i] = 0;
          
          U2n[i] = 1e-4 / params.CSolid;                        // H+     
          U3n[i] = 1e-4 / params.CSolid;                        // OH-
          U4n[i] = params.C0HCO3 / params.CSolid;               // HCO3-
          U5n[i] = 0.0;                                         // CO3 2-
          U6n[i] = params.C0HPO4 / params.CSolid;               // HPO4 2-     
          U7n[i] = 0.0;                                         // PO4 3-
          U8n[i] = params.C0Ca / params.CSolid;                 // Ca2+
          U9n[i] = params.C0Na / params.CSolid;                 // Na+
          U10n[i] = params.C0Cl / params.CSolid;                // Cl-     
          U11n[i] = 0.0;                                        // Phi
        } else {
          U1n[i] = params.CSolid / params.CSolid;               // Mg2+
          nodePhase[i] = 1;
          
          U2n[i] = 0.0;                                         // H+     
          U3n[i] = 0.0;                                         // OH-
          U4n[i] = 0.0;                                         // HCO3-
          U5n[i] = 0.0;                                         // CO3 2-
          U6n[i] = 0.0;                                         // HPO4 2-     
          U7n[i] = 0.0;                                         // PO4 3-
          U8n[i] = 0.0;                                         // Ca2+
          U9n[i] = 0.0;                                         // Na+
          U10n[i] = 0.0;                                        // Cl-     
          U11n[i] = 0.0;                                        // Phi
        }
        // Count number of solid nodes
        if (nodePhase[i] == 1) solidCount += 1;
      }
  }

  if (argc < 4) {
    utils::write_vector_paraview_scalar_to_file(
        JobName + Prefix + "-" +"0" + ".u1.csv", nodes, U1n, "U1n");
    utils::write_vector_paraview_scalar_to_file(
        JobName + Prefix + "-" + "0" + ".u2.csv", nodes, U2n, "U2n");
    utils::write_vector_paraview_scalar_to_file(
        JobName + Prefix + "-" + "0" + ".phi.csv", nodes, U11n, "phi");
    utils::write_vector_paraview_scalar_to_file(
        JobName + Prefix + "-" + "0" + ".phase.csv", nodes, nodePhase,
        "Phase");
  }

  double vol0mm = solidCount * params.dx * params.dy * params.dz * 1e9;
  std::cout << "Initial volume in mm^3: " << vol0mm << std::endl;

  double vol0rel = 0.0;
  double voln1rel = 0.0;
#pragma omp parallel for reduction(+ : vol0rel)
  for (int i = 0; i < params.nodesNo; i++) {
    if (nodePhase[i] == 1) vol0rel += (U1n[i] - params.CSat / params.CSolid);
  }

  // Construct global tangent and residual matrices
  SpMat sparseK1(params.nodesNo, params.nodesNo);
  SpMat sparseK2(params.nodesNo, params.nodesNo);
  SpMat sparseK3(params.nodesNo, params.nodesNo);
  SpMat sparseK4(params.nodesNo, params.nodesNo);
  SpMat sparseK5(params.nodesNo, params.nodesNo);
  SpMat sparseK6(params.nodesNo, params.nodesNo);
  SpMat sparseK7(params.nodesNo, params.nodesNo);
  SpMat sparseK8(params.nodesNo, params.nodesNo);
  SpMat sparseK9(params.nodesNo, params.nodesNo);
  SpMat sparseK10(params.nodesNo, params.nodesNo);
  SpMat sparseK11(params.nodesNo, params.nodesNo);

  //// Time marching loop ////

  // Clear energy and volume loss files
  energyFile.open(JobName + Prefix + ".energy",
                  std::ios::out | std::ios::trunc);
  energyFile.close();

  volumeLossFile.open(JobName + Prefix + ".volume",
                      std::ios::out | std::ios::trunc);
  volumeLossFile.close();

  // Restart from file
  std::stringstream n;
  if (argc == 4) {
    s0 = StartFileNo;
    n << s0;

    U1n.setZero();
    U2n.setZero();
    U3n.setZero();
    U4n.setZero();
    U5n.setZero();
    U6n.setZero();
    U7n.setZero();
    U8n.setZero();
    U9n.setZero();
    U10n.setZero();
    U11n.setZero();
    nodePhase.setZero();

    utils::read_vector_from_startfile(
        JobName + Prefix + "-" + n.str() + ".output.csv", params.nodesNo, nodes, U1n, "Mg++",  U2n, "H+",    U3n, "OH-", 
                                                                 U4n, "HCO3-", U5n, "CO32-", U6n, "HPO42-",
                                                                 U7n, "PO43-", U8n, "Ca++",  U9n, "Na+", U10n, "Cl-",
                                                                 U11n, "Phi",  nodePhase, "Phase");
    
    // Calculate Volume
    voln1rel = 0.0;
#pragma omp parallel for reduction(+ : voln1rel)
    for (int i = 0; i < params.nodesNo; i++) {
      if (nodePhase[i] == 1) voln1rel += (U1n[i] - params.CSat / params.CSolid);
    }
    voln1rel /= vol0rel;

    std::cout << "Volume in mm^3: " << solidCount * params.dx * params.dy * params.dz * 1e9
              << std::endl;
    std::cout << "Volume rel: " << voln1rel << std::endl;    
  }

  timer.tic();
  std::cout << "Time Marching started:" << std::endl;

  for (int s = s0; s <= params.stepsNo; s++) {
    if ((s % sub_step_no == 0) && (s > s0)) {
      std::cout << "Time for a step: " << timer.toc() << " seconds."
                << std::endl;
      timer.tic();
    }

    const double t = s * params.dt;

    if (s == s0) {
      U1n1 = U1n;
      U2n1 = U2n;
      U3n1 = U3n;
      U4n1 = U4n;
      U5n1 = U5n;
      U6n1 = U6n;
      U7n1 = U7n;
      U8n1 = U8n;
      U9n1 = U9n;
      U10n1 = U10n;
      U11n1 = U11n;

#pragma omp parallel for
      for (int i = 0; i < params.nodesNo; i++) {
        if (boundaryConditionTypes[i] == 0) {
          if (boundaryConditionValues1[i] != -1)
            U1n1[i] = boundaryConditionValues1[i] / params.CSolid;
          else
            U1n1[i] = U1n[i];

          if (boundaryConditionValues2[i] != -1)
            U2n1[i] = boundaryConditionValues2[i] / params.CSolid;
          else
            U2n1[i] = U2n[i];

          if (boundaryConditionValues3[i] != -1)
            U3n1[i] = boundaryConditionValues3[i] / params.CSolid;
          else
            U3n1[i] = U3n[i];

          if (boundaryConditionValues4[i] != -1)
            U4n1[i] = boundaryConditionValues4[i] / params.CSolid;
          else
            U4n1[i] = U4n[i];

          if (boundaryConditionValues5[i] != -1)
            U5n1[i] = boundaryConditionValues5[i] / params.CSolid;
          else
            U5n1[i] = U5n[i];

          if (boundaryConditionValues6[i] != -1)
            U6n1[i] = boundaryConditionValues6[i] / params.CSolid;
          else
            U6n1[i] = U6n[i];

          if (boundaryConditionValues7[i] != -1)
            U7n1[i] = boundaryConditionValues7[i] / params.CSolid;
          else
            U7n1[i] = U7n[i];

          if (boundaryConditionValues8[i] != -1)
            U8n1[i] = boundaryConditionValues8[i] / params.CSolid;
          else
            U8n1[i] = U8n[i];  

          if (boundaryConditionValues9[i] != -1)
            U9n1[i] = boundaryConditionValues9[i] / params.CSolid;
          else
            U9n1[i] = U9n[i];

          if (boundaryConditionValues10[i] != -1)
            U10n1[i] = boundaryConditionValues10[i] / params.CSolid;
          else
            U10n1[i] = U10n[i];

          if (boundaryConditionValues11[i] != -1)
            U11n1[i] = boundaryConditionValues11[i];
          else
            U11n1[i] = U11n[i];
        } else {
          U1n1[i] = U1n[i];
          U2n1[i] = U2n[i];
          U3n1[i] = U3n[i];
          U4n1[i] = U4n[i];
          U5n1[i] = U5n[i];
          U6n1[i] = U6n[i];
          U7n1[i] = U7n[i];
          U8n1[i] = U8n[i];
          U9n1[i] = U9n[i];
          U10n1[i] = U10n[i];
          U11n1[i] = U11n[i];
        }
      }
    }

    res1Old = 0.0;
    res2Old = 0.0;
    res3Old = 0.0;
    res4Old = 0.0;
    res5Old = 0.0;
    res6Old = 0.0;
    res7Old = 0.0;
    res8Old = 0.0;
    res9Old = 0.0;
    res10Old = 0.0;
    res11Old = 0.0;

    resMax = 0.0;

    // Update the tangent stiffness matrix
    utils::NewtonMatrices(sparseK1, sparseK2, sparseK3, sparseK4, sparseK5, sparseK6, sparseK7, sparseK8, sparseK9, sparseK10, sparseK11,
                          nodes, horizons, boundaryConditionValues1, boundaryConditionValues2, boundaryConditionValues3, boundaryConditionValues4,
                          boundaryConditionValues5, boundaryConditionValues6, boundaryConditionValues7, boundaryConditionValues8, boundaryConditionValues9,
                          boundaryConditionValues10, boundaryConditionValues11, U1n1, U2n1, U3n1, U4n1, U5n1, U6n1, U7n1, U8n1, U9n1, U10n1, U11n1,
                          U1n, U2n, U3n, U4n, U5n, U6n, U7n, U8n, U9n, U10n, U11n,
                          nodePhase, horizonsLengths, boundaryConditionTypes, voln1rel, params);

    // Newton iteration
    for (int i = 0; i < MAX_ITER; i++) {
      // Calculate residuals
      utils::Residuals(nodes, horizons, boundaryConditionValues1, boundaryConditionValues2, boundaryConditionValues3,
                      boundaryConditionValues4, boundaryConditionValues5, boundaryConditionValues6, boundaryConditionValues7,
                      boundaryConditionValues8, boundaryConditionValues9, boundaryConditionValues10, boundaryConditionValues11,
                      U1n1, U2n1, U3n1, U4n1, U5n1, U6n1, U7n1, U8n1, U9n1, U10n1, U11n1,
                      U1n, U2n, U3n, U4n, U5n, U6n, U7n, U8n, U9n, U10n, U11n,
                      U1n0, U2n0, U3n0, U4n0, U5n0, U6n0, U7n0, U8n0, U9n0, U10n0, U11n0,
                      F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, nodePhase, horizonsLengths, boundaryConditionTypes, voln1rel, params);


#pragma omp parallel for
      for (int i = 0; i < params.nodesNo; i++) {
        if (boundaryConditionTypes[i] == 0) {
          if (boundaryConditionValues1[i] != -1)
            F1[i] = U1n1[i] - boundaryConditionValues1[i] / params.CSolid;

          if (boundaryConditionValues2[i] != -1)
            F2[i] = U2n1[i] - boundaryConditionValues2[i] / params.CSolid;

          if (boundaryConditionValues3[i] != -1)
            F3[i] = U3n1[i] - boundaryConditionValues3[i] / params.CSolid;

          if (boundaryConditionValues4[i] != -1)
            F4[i] = U4n1[i] - boundaryConditionValues4[i] / params.CSolid;

          if (boundaryConditionValues5[i] != -1)
            F5[i] = U5n1[i] - boundaryConditionValues5[i] / params.CSolid;

          if (boundaryConditionValues6[i] != -1)
            F6[i] = U6n1[i] - boundaryConditionValues6[i] / params.CSolid;

          if (boundaryConditionValues7[i] != -1)
            F7[i] = U7n1[i] - boundaryConditionValues7[i] / params.CSolid;

          if (boundaryConditionValues8[i] != -1)
            F8[i] = U8n1[i] - boundaryConditionValues8[i] / params.CSolid;
          
          if (boundaryConditionValues9[i] != -1)
            F9[i] = U9n1[i] - boundaryConditionValues9[i] / params.CSolid;

          if (boundaryConditionValues10[i] != -1)
            F10[i] = U10n1[i] - boundaryConditionValues10[i] / params.CSolid;

          if (boundaryConditionValues11[i] != -1)
            F11[i] = U11n1[i] - boundaryConditionValues11[i];
        }
      }

      Eigen::GMRES<SpMat, Eigen::DiagonalPreconditioner<double>> solver1(
          sparseK1);
      solver1.setTolerance(EPS);
      solver1.setMaxIterations(MAX_ITER);

      Eigen::GMRES<SpMat, Eigen::DiagonalPreconditioner<double>> solver2(
          sparseK2);
      solver2.setTolerance(EPS);
      solver2.setMaxIterations(MAX_ITER);

      Eigen::GMRES<SpMat, Eigen::DiagonalPreconditioner<double>> solver3(
          sparseK3);
      solver3.setTolerance(EPS);
      solver3.setMaxIterations(MAX_ITER);

      Eigen::GMRES<SpMat, Eigen::DiagonalPreconditioner<double>> solver4(
          sparseK4);
      solver4.setTolerance(EPS);
      solver4.setMaxIterations(MAX_ITER);

      Eigen::GMRES<SpMat, Eigen::DiagonalPreconditioner<double>> solver5(
          sparseK5);
      solver5.setTolerance(EPS);
      solver5.setMaxIterations(MAX_ITER);

      Eigen::GMRES<SpMat, Eigen::DiagonalPreconditioner<double>> solver6(
          sparseK6);
      solver6.setTolerance(EPS);
      solver6.setMaxIterations(MAX_ITER);

      Eigen::GMRES<SpMat, Eigen::DiagonalPreconditioner<double>> solver7(
          sparseK7);
      solver7.setTolerance(EPS);
      solver7.setMaxIterations(MAX_ITER);

      Eigen::GMRES<SpMat, Eigen::DiagonalPreconditioner<double>> solver8(
          sparseK8);
      solver8.setTolerance(EPS);
      solver8.setMaxIterations(MAX_ITER);

      Eigen::GMRES<SpMat, Eigen::DiagonalPreconditioner<double>> solver9(
          sparseK9);
      solver9.setTolerance(EPS);
      solver9.setMaxIterations(MAX_ITER);

      Eigen::GMRES<SpMat, Eigen::DiagonalPreconditioner<double>> solver10(
          sparseK10);
      solver10.setTolerance(EPS);
      solver10.setMaxIterations(MAX_ITER);

      Eigen::GMRES<SpMat, Eigen::DiagonalPreconditioner<double>> solver11(
          sparseK11);
      solver11.setTolerance(EPS);
      solver11.setMaxIterations(MAX_ITER);

      dU1 = solver1.solve(-F1);
      dU2 = solver2.solve(-F2);
      dU3 = solver3.solve(-F3);
      dU4 = solver4.solve(-F4);
      dU5 = solver5.solve(-F5);
      dU6 = solver6.solve(-F6);
      dU7 = solver7.solve(-F7);
      dU8 = solver8.solve(-F8);
      dU9 = solver9.solve(-F9);
      dU10 = solver10.solve(-F10);
      dU11 = solver11.solve(-F11);

      U1n1 = U1n1 + dU1;
      U2n1 = U2n1 + dU2;
      U3n1 = U3n1 + dU3;
      U4n1 = U4n1 + dU4;
      U5n1 = U5n1 + dU5;
      U6n1 = U6n1 + dU6;
      U7n1 = U7n1 + dU7;
      U8n1 = U8n1 + dU8;
      U9n1 = U9n1 + dU9;
      U10n1 = U10n1 + dU10;
      U11n1 = U11n1 + dU11;

      res1 = dU1.norm();
      res2 = dU2.norm();
      res3 = dU3.norm();
      res4 = dU4.norm();
      res5 = dU5.norm();
      res6 = dU6.norm();
      res7 = dU7.norm();
      res8 = dU8.norm();
      res9 = dU9.norm();
      res10 = dU10.norm();
      res11 = dU11.norm();

      #pragma omp parallel for
      for (int i = 0; i < params.nodesNo; i++) {
         if (U1n1[i] < 1.0e-8 / params.CSolid) {   
             U1n1[i] = 1.0e-8 / params.CSolid;
         }
         if (U2n1[i] < 1.0e-8 / params.CSolid) {   
             U2n1[i] = 1.0e-8 / params.CSolid;
         }
         if (U3n1[i] < 1.0e-8 / params.CSolid) {   
             U3n1[i] = 1.0e-8 / params.CSolid;
         }
         if (U4n1[i] < 1.0e-8 / params.CSolid) {   
             U4n1[i] = 1.0e-8 / params.CSolid;
         }
         if (U5n1[i] < 1.0e-8 / params.CSolid) {   
             U5n1[i] = 1.0e-8 / params.CSolid;
         }
         if (U6n1[i] < 1.0e-8 / params.CSolid) {   
             U6n1[i] = 1.0e-8 / params.CSolid;
         }
         if (U7n1[i] < 1.0e-8 / params.CSolid) {   
             U7n1[i] = 1.0e-8 / params.CSolid;
         }
         if (U8n1[i] < 1.0e-8 / params.CSolid) {   
             U8n1[i] = 1.0e-8 / params.CSolid;
         }
         if (U9n1[i] < 1.0e-8 / params.CSolid) {   
             U9n1[i] = 1.0e-8 / params.CSolid; 
         }
         if (U10n1[i] < 1.0e-8 / params.CSolid && nodePhase[i] == 0) {   
             U10n1[i] = 1.0e-8 / params.CSolid; 
         }
      }

      //      res1 = 0.0;
      //      res2 = 0.0;
      //      res3 = 0.0;
      //      res4 = 0.0;
      //      res5 = 0.0;
      //      res6 = 0.0;
      //      res7 = 0.0;
      //      res8 = 0.0;
      //      res9 = 0.0;
      //      res10 = 0.0;
      //      res11 = 0.0;

      //      res1Old = 0.0;
      //      res2Old = 0.0;
      //      res3Old = 0.0;
      //      res4Old = 0.0;
      //      res5Old = 0.0;
      //      res6Old = 0.0;
      //      res7Old = 0.0;
      //      res8Old = 0.0;
      //      res9Old = 0.0;
      //      res10Old = 0.0;
      //      res11Old = 0.0;

      resMax = std::max(std::max(std::max(std::max(std::max(std::max(std::max(std::max(std::max(std::max(
                        std::abs(res1 - res1Old),
                        std::abs(res2 - res2Old)),
                        std::abs(res3 - res3Old)),
                        std::abs(res4 - res4Old)),
                        std::abs(res5 - res5Old)),
                        std::abs(res6 - res6Old)),
                        std::abs(res7 - res7Old)),
                        std::abs(res8 - res8Old)),
                        std::abs(res9 - res9Old)),
                        std::abs(res10 - res10Old)),
                        std::abs(res11 - res11Old));

      if (resMax < EPS) {
        // if (s % sub_step_no == 0) {
          std::cout << "\n\tres1: " << res1 << " res2: " << res2
                    << " res3: " << res3 << " res4: " << res4
                    << " res5: " << res5 << " res6: " << res6 
                    << " res7: " << res7 << " res8: " << res8 
                    << " res9: " << res9 << " res10: " << res10
                    << " res11: " << res11 << std::endl;
          std::cout << "\n\t#outer iterations: " << i << std::endl;
          std::cout << "\tnorm of residual: " << resMax << std::endl;
        // }
        break;
      } else {
        res1Old = res1;
        res2Old = res2;
        res3Old = res3;
        res4Old = res4;
        res5Old = res5;
        res6Old = res6;
        res7Old = res7;
        res8Old = res8;
        res9Old = res9;
        res10Old = res10;
        res11Old = res11;

        // if (s % sub_step_no == 0) {
          std::cout << "\n\tres1: " << res1 << " res2: " << res2
                    << " res3: " << res3 << " res4: " << res4
                    << " res5: " << res5 << " res6: " << res6 
                    << " res7: " << res7 << " res8: " << res8 
                    << " res9: " << res9 << " res10: " << res10
                    << " res11: " << res11 << std::endl;
          std::cout << "\n\t#outer iteration: " << i << std::endl;
          std::cout << "\tnorm of residual: " << resMax << std::endl;
        // }
      }
    }

    // Update number of solid nodes for volume calculation
#pragma omp parallel for reduction(- : solidCount)
    for (int i = 0; i < params.nodesNo; i++) {
      if (nodePhase[i] == 1 && U1n1[i] <= params.CSat / params.CSolid) {
        nodePhase[i] = 0;

        solidCount -= 1;
      }
    }

    if (solidCount <= 0) {
      std::cout << "Solid phase corroded completely at t: " << t
                << " seconds or " << std::ceil(static_cast<double>(t) / 60.)
                << " minutes or "
                << std::ceil(static_cast<double>(t) / 60. / 60.) << " hours or "
                << std::ceil(static_cast<double>(t) / 60. / 60. / 24.)
                << " days." << std::endl;
      return 0;
    }

    // Save needed results
    if (s % sub_step_no == 0) {
      std::cout << "\nt: " << t
                << " seconds or "
                << std::ceil(static_cast<double>(t) / 60.) << " minutes or "
                << std::ceil(static_cast<double>(t) / 60. / 60.) << " hours or "
                << std::ceil(static_cast<double>(t) / 60. / 60. / 24.)
                << " days." << std::endl;

      // Calculate Volume
      voln1rel = 0.0;
#pragma omp parallel for reduction(+ : voln1rel)
      for (int i = 0; i < params.nodesNo; i++) {
        if (nodePhase[i] == 1) voln1rel += (U1n1[i] - params.CSat / params.CSolid);
      }
      voln1rel /= vol0rel;

      std::cout << "Volume in mm^2: " << solidCount * params.dx * params.dy * params.dz * 1e9
                << std::endl;
      std::cout << "Volume rel: " << voln1rel << std::endl;

      // Calculate volume
      volumeLossFile.precision(17);
      volumeLossFile.open(JobName + Prefix + ".volume",
                          std::ios::out | std::ios::app);
      volumeLossFile << voln1rel << "," << std::endl;
      volumeLossFile.close();

      // Calculate Energy
      energyValue = 0.0;
#pragma omp parallel for reduction(+ : energyValue)
      for (int i = 0; i < params.nodesNo; i++) {
        if ((std::abs(nodes(i, 0)) < energyBound) ||
            (std::abs(nodes(i, 1)) < energyBound)) {
          energyValue += U1n1[i] * U1n1[i] * params.dx * params.dy * params.dz;
        }
      }

      std::cout << "Energy: " << energyValue << std::endl;

      std::stringstream n;
      if (argc < 4) n << s;
      else n << (s + StartFileNo);

      /* utils::write_vector_paraview_scalar_to_file(
          JobName + Prefix + "-" + n.str() + ".u1.csv", nodes, U1n1, "U1n");
      utils::write_vector_paraview_scalar_to_file(
          JobName + Prefix + "-" + n.str() + ".u2.csv", nodes, U2n1, "U2n");
      utils::write_vector_paraview_scalar_to_file(
          JobName + Prefix + "-" + n.str() + ".phi.csv", nodes, U11n1, "phi");
      utils::write_vector_paraview_scalar_to_file(
          JobName + Prefix + "-" + n.str() + ".phase.csv", nodes, nodePhase,
          "Phase");
      */

      utils::write_vector_paraview_scalar_to_file(
          JobName + Prefix + "-" + n.str() + ".output.csv", nodes, U1n1, "Mg++", U2n1, "H+", U3n1, "OH-", 
                                                                  U4n1, "HCO3-", U5n1, "CO32-", U6n1, "HPO42-",
                                                                  U7n1, "PO43-", U8n1, "Ca++", U9n1, "Na+", U10n1, "Cl-",
                                                                  U11n1, "Phi", nodePhase, "Phase");

      energyFile.open(JobName + Prefix + ".energy",
                      std::ios::out | std::ios::app);
      energyFile << energyValue << std::endl;
      energyFile.close();
    }

    // Prepare for next time step
    U1n0.swap(U1n);
    U2n0.swap(U2n);
    U3n0.swap(U3n);
    U4n0.swap(U4n);
    U5n0.swap(U5n);
    U6n0.swap(U6n);
    U7n0.swap(U7n);
    U8n0.swap(U8n);
    U9n0.swap(U9n);
    U10n0.swap(U10n);
    U11n0.swap(U11n);

    U1n.swap(U1n1);
    U2n.swap(U2n1);
    U3n.swap(U3n1);
    U4n.swap(U4n1);
    U5n.swap(U5n1);
    U6n.swap(U6n1);
    U7n.swap(U7n1);
    U8n.swap(U8n1);
    U9n.swap(U9n1);
    U10n.swap(U10n1);
    U11n.swap(U11n1);
  }

  std::cout << "Time marching took " << timer.toc() << " seconds." << std::endl;
  return 0;
}
