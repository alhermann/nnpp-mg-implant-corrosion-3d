#include <math.h>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN  // Faster build process
#define NOMINMAX
#include <Windows.h>
#else
#include <sys/time.h>
#endif

namespace utils {

// Beta correction function
double Beta(double xi, double delta, double dx) {
  if (xi <= delta - dx / 2.0) {
    return 1.0;
  } else if ((xi > delta - dx / 2.0) && (xi <= delta + dx / 2.0)) {
    return (delta + dx / 2.0 - xi) / (dx);
  } else {
    return 0.0;
  }
}

template <typename vector_Itype, typename vector_Dtype, typename matrix_Itype,
          typename matrix_Dtype, typename parameters>
void Residuals(
    matrix_Dtype &nodes, matrix_Itype &horizons,
    vector_Dtype &boundaryConditionValues1,
    vector_Dtype &boundaryConditionValues2,
    vector_Dtype &boundaryConditionValues3,
    vector_Dtype &boundaryConditionValues4,
    vector_Dtype &boundaryConditionValues5, 
    vector_Dtype &boundaryConditionValues6,
    vector_Dtype &boundaryConditionValues7,
    vector_Dtype &boundaryConditionValues8,
    vector_Dtype &boundaryConditionValues9,
    vector_Dtype &boundaryConditionValues10,
    vector_Dtype &boundaryConditionValues11,
    vector_Dtype &U1n1, vector_Dtype &U2n1, vector_Dtype &U3n1, vector_Dtype &U4n1,
    vector_Dtype &U5n1, vector_Dtype &U6n1, vector_Dtype &U7n1, vector_Dtype &U8n1, 
    vector_Dtype &U9n1, vector_Dtype &U10n1,vector_Dtype &U11n1,
    vector_Dtype &U1n, vector_Dtype &U2n, vector_Dtype &U3n, vector_Dtype &U4n,
    vector_Dtype &U5n, vector_Dtype &U6n, vector_Dtype &U7n, vector_Dtype &U8n, 
    vector_Dtype &U9n, vector_Dtype &U10n,vector_Dtype &U11n,
    vector_Dtype &U1n0, vector_Dtype &U2n0, vector_Dtype &U3n0, vector_Dtype &U4n0,
    vector_Dtype &U5n0, vector_Dtype &U6n0, vector_Dtype &U7n0, vector_Dtype &U8n0, 
    vector_Dtype &U9n0, vector_Dtype &U10n0,vector_Dtype &U11n0,
    vector_Dtype &F1, vector_Dtype &F2, vector_Dtype &F3, vector_Dtype &F4, vector_Dtype &F5, 
    vector_Dtype &F6, vector_Dtype &F7, vector_Dtype &F8, vector_Dtype &F9, vector_Dtype &F10, 
    vector_Dtype &F11, 
    vector_Itype &nodePhase,
    vector_Itype &horizonsLengths, vector_Itype &boundaryConditionTypes,
    double voln1rel, parameters &params) {
  F1.setZero();
  F2.setZero();
  F3.setZero();
  F4.setZero();
  F5.setZero();
  F6.setZero();
  F7.setZero();
  F8.setZero();
  F9.setZero();
  F10.setZero();
  F11.setZero();

  // Mg
#pragma omp declare reduction(merge : std::vector <Eigen::Triplet <double>>: omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for
  for (int i = 0; i < params.nodesNo; i++) {
    double R1 = 0.0;
    double sum_KR11 = 0.0;
    double sum_KR12 = 0.0;
    double phi_c = 0.0;
    double BV = 1.0;
    double eta = 0.0;
    int interfaceFlag = 0;

    // Interface Nodes
    if (nodePhase[i] == 1)
      for (int j = 1; j < horizonsLengths[i]; j++) {
        int n = horizons(i, j);
        if (nodePhase[n] == 0) {
          interfaceFlag = 1;
          break;
        }
      }

    for (int j = 1; j < horizonsLengths[i]; j++) {
      int n = horizons(i, j);

      double xi_x = nodes(n, 0) - nodes(i, 0);
      double xi_y = nodes(n, 1) - nodes(i, 1);
      double xi_z = nodes(n, 2) - nodes(i, 2);

      double xi = std::sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);

      if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        /*phi_c = (params.R * params.T) / (params.F * params.z1) * std::log(U1n[i] / (params.CSat / params.CSolid));

        eta = params.phim + params.phimse - U11n[n] - phi_c;

        BV = std::exp(params.alphaA * params.z1 * params.F * eta / (params.R * params.T)); */
        BV = std::pow(10., -params.vol_red * (1. - voln1rel));
      } else if (nodePhase[n] == 1 && nodePhase[i] == 0) {
        /*phi_c = (params.R * params.T) / (params.F * params.z1) * std::log(U1n[n] / (params.CSat / params.CSolid));

        eta = params.phim + params.phimse - U11n[i] - phi_c;

        BV = std::exp(params.alphaA * params.z1 * params.F * eta / (params.R * params.T));*/
        BV = std::pow(10., -params.vol_red * (1. - voln1rel));
      } else {
        BV = 1.0;
      }

      // BV = 1.0;

      if (nodePhase[i] == 1 && nodePhase[n] == 1) {
        // continue;
        sum_KR11 += -(params.K1prop1 * params.D1Solid * BV * (U1n1[n] - U1n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s11)) *
                      utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR12 += -(params.K1prop2 * params.D1Solid * BV * params.z1 * params.F * U1n1[n] * (U11n[n] - U11n[i]) /
                      (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s12))) *
                      utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

      } else if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        sum_KR11 += -(params.K1prop1 * params.D1Solid * BV * (U1n1[n] - U1n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s11)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR12 +=
            -(params.K1prop2 * params.D1Solid * BV * params.z1 * params.F * U1n1[n] * (U11n[n] - U11n[i]) /
              (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s12))) *
            utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

      } else if (nodePhase[i] == 0 && nodePhase[n] == 1) {
        sum_KR11 += -(params.K1prop1 * params.D1Solid * BV * (U1n1[n] - U1n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s11)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR12 +=
            -(params.K1prop2 * params.D1Solid * BV * params.z1 * params.F * U1n1[n] * (U11n[n] - U11n[i]) /
              (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s12))) *
            utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

      } else {
        sum_KR11 += -(params.K1prop1 * params.D1Liquid * (U1n1[n] - U1n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s11)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR12 +=
            -(params.K1prop2 * params.D1Liquid * params.z1 * params.F * U1n1[n] * (U11n[n] - U11n[i]) /
              (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s12))) *
            utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;
      }
    }

    if (/*nodePhase[i] == 1 &&*/ interfaceFlag == 1)
      R1 =  -params.k1b * std::max((U1n1[i] * U3n[i] * U3n[i] - std::pow(10., -params.pKMgOH2) / params.CSolid), 0.0)
            -params.k2b * std::max((U1n1[i] * U5n[i] - std::pow(10., -params.pKMgCO3) / params.CSolid), 0.0)  
            -params.k3b * std::max((U1n1[i] * U5n[i] - std::pow(10., -params.pKMgHCO3) / params.CSolid), 0.0)
            -params.k4b * std::max((U1n[i] * U1n[i] * U1n[i] * U6n[i] * U6n[i] - std::pow(10., -params.pKMgPO4) / params.CSolid), 0.0);

    else if (nodePhase[i] == 0)
      R1 =  -params.k1b * std::max((U1n1[i] * U3n[i] * U3n[i] - std::pow(10., -params.pKMgOH2) / params.CSolid), 0.0)
            -params.k2b * std::max((U1n1[i] * U5n[i] - std::pow(10., -params.pKMgCO3) / params.CSolid), 0.0)  
            -params.k3b * std::max((U1n1[i] * U5n[i] - std::pow(10., -params.pKMgHCO3) / params.CSolid), 0.0)
            -params.k4b * std::max((U1n[i] * U1n[i] * U1n[i] * U6n[i] * U6n[i] - std::pow(10., -params.pKMgPO4) / params.CSolid), 0.0);
    else
      R1 = 0.0;

    // Residual
    F1[i] = ((U1n1[i] - U1n[i]) / params.dt) + sum_KR11 + sum_KR12 - R1;
  }

// H+
//#pragma omp declare reduction(merge : std::vector <Eigen::Triplet <double>>:
// omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for
  for (int i = 0; i < params.nodesNo; i++) {
    double R2 = 0.0;
    double sum_KR21 = 0.0;
    double sum_KR22 = 0.0;
    double eta = 0.0;
    double phic = 0.0;
    int interfaceFlag = 0;

    // Interface Nodes
    if (nodePhase[i] == 1)
      for (int j = 1; j < horizonsLengths[i]; j++) {
        int n = horizons(i, j);
        if (nodePhase[n] == 0) {
          interfaceFlag = 1;
          break;
        }
      }

    for (int j = 1; j < horizonsLengths[i]; j++) {
      int n = horizons(i, j);

      double xi_x = nodes(n, 0) - nodes(i, 0);
      double xi_y = nodes(n, 1) - nodes(i, 1);
      double xi_z = nodes(n, 2) - nodes(i, 2);

      double xi = std::sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);

      if (nodePhase[i] == 1 && nodePhase[n] == 1) {
        continue;
      } else if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        sum_KR21 += -(params.K2prop1 * params.D2Solid * (U2n1[n] - U2n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s21)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR22 += -(params.K2prop2 * params.D2Solid * params.z2 * params.F * U2n1[n] * (U11n[n] - U11n[i]) /
                      (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s22))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;
      } else if (nodePhase[i] == 0 && nodePhase[n] == 1) {
        sum_KR21 += -(params.K2prop1 * params.D2Solid * (U2n1[n] - U2n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s21)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR22 += -(params.K2prop2 * params.D2Solid * params.z2 * params.F * U2n1[n] * (U11n[n] - U11n[i]) /
                      (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s22))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;
      } else {
        sum_KR21 += -(params.K2prop1 * params.D2Liquid * (U2n1[n] - U2n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s21)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR22 +=
            -(params.K2prop2 * params.D2Liquid * params.z2 * params.F * U2n1[n] * (U11n[n] - U11n[i]) /
              (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s22))) *
            utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;
      }
    }

    if (/*nodePhase[i] == 1 &&*/ interfaceFlag == 1) {
      phic = (params.R * params.T / (params.F * params.z1)) * std::log(U1n[i] / (params.CSat / params.CSolid));
      eta = params.phim + params.phimse - U11n[i] - phic;

      R2 += (params.JHp / (params.z2 * params.F * params.CSolid)) * std::exp(-params.alphaC * params.F * eta / (params.R * params.T));
    } 
    else if (nodePhase[i] == 0){
      R2 =   params.k7b * (std::pow(10., -params.pKH20) / params.CSolid  - U2n1[i] * U3n[i])
           + params.k8b * (std::pow(10., -params.pKHCO3) / params.CSolid - U2n1[i] * U5n[i])
           + params.k9b * (std::pow(10., -params.pKHPO4) / params.CSolid - U2n1[i] * U7n[i]); 
    }
    else
      R2 = 0.0;

    // Residual
    F2[i] = ((U2n1[i] - U2n[i]) / params.dt) + sum_KR21 + sum_KR22 - R2;
  }

// OH-
//#pragma omp declare reduction(merge : std::vector <Eigen::Triplet <double>>:
// omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for
  for (int i = 0; i < params.nodesNo; i++) {
    double R3 = 0.0;
    double sum_KR31 = 0.0;
    double sum_KR32 = 0.0;
    double eta = 0.0;
    double phic = 0.0;
    int interfaceFlag = 0;

    // Interface Nodes
    if (nodePhase[i] == 1)
      for (int j = 1; j < horizonsLengths[i]; j++) {
        int n = horizons(i, j);
        if (nodePhase[n] == 0) {
          interfaceFlag = 1;
          break;
        }
      }

    for (int j = 1; j < horizonsLengths[i]; j++) {
      int n = horizons(i, j);

      double xi_x = nodes(n, 0) - nodes(i, 0);
      double xi_y = nodes(n, 1) - nodes(i, 1);
      double xi_z = nodes(n, 2) - nodes(i, 2);

      double xi = std::sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);

      if (nodePhase[i] == 1 && nodePhase[n] == 1) {
        continue;
      } else if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        sum_KR31 += -(params.K3prop1 * params.D3Solid * (U3n1[n] - U3n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s31)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR32 += -(params.K3prop2 * params.D3Solid * params.z3 * params.F * U3n1[n] * (U11n[n] - U11n[i]) /
                      (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s32))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

      } else if (nodePhase[i] == 0 && nodePhase[n] == 1) {
        sum_KR31 += -(params.K3prop1 * params.D3Solid * (U3n1[n] - U3n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s31)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR32 += -(params.K3prop2 * params.D3Solid * params.z3 * params.F * U3n1[n] * (U11n[n] - U11n[i]) /
                      (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s32))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;
      } else {
        sum_KR31 += -(params.K3prop1 * params.D3Liquid * (U3n1[n] - U3n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s31)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR32 +=
            -(params.K3prop2 * params.D3Liquid * params.z3 * params.F * U3n1[n] * (U11n[n] - U11n[i]) /
              (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s32))) *
            utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;
      }
    }

    if (/*nodePhase[i] == 1 &&*/ interfaceFlag == 1) {
      R3 = - params.k1b * std::max((U3n[i] * U3n[i] * U1n[i] - std::pow(10., -params.pKMgOH2) / params.CSolid), 0.0)
           - params.k6b * std::max((U3n1[i] * U8n[i] * U8n[i] * U8n[i] * U8n[i] * U8n[i] * U7n[i] * U7n[i] * U7n[i] - std::pow(10., -params.pKCa5OHPO4) / params.CSolid), 0.0);
    }
    else if (nodePhase[i] == 0) {
      R3 = - params.k1b * std::max((U3n[i] * U3n[i] * U1n[i] - std::pow(10., -params.pKMgOH2) / params.CSolid), 0.0)
           - params.k6b * std::max((U3n1[i] * U8n[i] * U8n[i] * U8n[i] * U8n[i] * U8n[i] * U7n[i] * U7n[i] * U7n[i] - std::pow(10., -params.pKCa5OHPO4) / params.CSolid), 0.0)
           + params.k7b * (std::pow(10., -params.pKH20) / params.CSolid - U3n1[i] * U2n[i]);
    } else
      R3 = 0.0;

    // Residual
    F3[i] = ((U3n1[i] - U3n[i]) / params.dt) + sum_KR31 + sum_KR32 - R3;
  }

// HCO3-
//#pragma omp declare reduction(merge : std::vector <Eigen::Triplet <double>>:
// omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for
  for (int i = 0; i < params.nodesNo; i++) {
    double R4 = 0.0;
    double sum_KR41 = 0.0;
    double sum_KR42 = 0.0;
    int interfaceFlag = 0;

    // Interface Nodes
    if (nodePhase[i] == 1)
      for (int j = 1; j < horizonsLengths[i]; j++) {
        int n = horizons(i, j);
        if (nodePhase[n] == 0) {
          interfaceFlag = 1;
          break;
        }
      }

    for (int j = 1; j < horizonsLengths[i]; j++) {
      int n = horizons(i, j);

      double xi_x = nodes(n, 0) - nodes(i, 0);
      double xi_y = nodes(n, 1) - nodes(i, 1);
      double xi_z = nodes(n, 2) - nodes(i, 2);

      double xi = std::sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);

      if (nodePhase[i] == 1 && nodePhase[n] == 1) {
        continue;
      } else if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        sum_KR41 += -(params.K4prop1 * params.D4Solid * (U4n1[n] - U4n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s41)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR42 += -(params.K4prop2 * params.D4Solid * params.z4 * params.F * U4n1[n] * (U11n[n] - U11n[i]) /
                      (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s42))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

      } else if (nodePhase[i] == 0 && nodePhase[n] == 1) {
        sum_KR41 += -(params.K4prop1 * params.D4Solid * (U4n1[n] - U4n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s41)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR42 += -(params.K4prop2 * params.D4Solid * params.z4 * params.F * U4n1[n] * (U11n[n] - U11n[i]) /
                      (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s42))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;
      } else {
        sum_KR41 += -(params.K4prop1 * params.D4Liquid * (U4n1[n] - U4n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s41)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR42 +=
            -(params.K4prop2 * params.D4Liquid * params.z4 * params.F * U4n1[n] * (U11n[n] - U11n[i]) /
              (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s42))) *
            utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;
      }
    }

    if (nodePhase[i] == 1 && interfaceFlag == 1)
      R4 = 0.0;
    else if (nodePhase[i] == 0)
      R4 = params.k8b * (std::pow(10., -params.pKHCO3) / params.CSolid - U4n1[i] * U2n[i]);
    else
      R4 = 0.0;

    // Residual
    F4[i] = ((U4n1[i] - U4n[i]) / params.dt) + sum_KR41 + sum_KR42 - R4;
  }

// CO3 2-
//#pragma omp declare reduction(merge : std::vector <Eigen::Triplet <double>>:
// omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for
  for (int i = 0; i < params.nodesNo; i++) {
    double R5 = 0.0;
    double sum_KR51 = 0.0;
    double sum_KR52 = 0.0;
    int interfaceFlag = 0;

    // Interface Nodes
    if (nodePhase[i] == 1)
      for (int j = 1; j < horizonsLengths[i]; j++) {
        int n = horizons(i, j);
        if (nodePhase[n] == 0) {
          interfaceFlag = 1;
          break;
        }
      }

    for (int j = 1; j < horizonsLengths[i]; j++) {
      int n = horizons(i, j);

      double xi_x = nodes(n, 0) - nodes(i, 0);
      double xi_y = nodes(n, 1) - nodes(i, 1);
      double xi_z = nodes(n, 2) - nodes(i, 2);

      double xi = std::sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);

      if (nodePhase[i] == 1 && nodePhase[n] == 1) {
        continue;
      } else if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        sum_KR51 += -(params.K5prop1 * params.D5Solid * (U5n1[n] - U5n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s51)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR52 += -(params.K5prop2 * params.D5Solid * params.z5 * params.F * U5n1[n] * (U11n[n] - U11n[i]) /
                      (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s52))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

      } else if (nodePhase[i] == 0 && nodePhase[n] == 1) {
        sum_KR51 += -(params.K5prop1 * params.D5Solid * (U5n1[n] - U5n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s51)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR52 += -(params.K5prop2 * params.D5Solid * params.z5 * params.F * U5n1[n] * (U11n[n] - U11n[i]) /
                      (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s52))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;
      } else {
        sum_KR51 += -(params.K5prop1 * params.D5Liquid * (U5n1[n] - U5n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s51)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR52 += -(params.K5prop2 * params.D5Liquid * params.z5 * params.F * U5n1[n] * (U11n[n] - U11n[i]) /
                      (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s52))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;
      }
    }

    if (nodePhase[i] == 1 && interfaceFlag == 1)
      R5 = -params.k2b * std::max((U5n1[i] * U1n[i] - std::pow(10., -params.pKMgCO3) / params.CSolid), 0.0)
           -params.k3b * std::max((U5n1[i] * U1n[i] - std::pow(10., -params.pKMgHCO3) / params.CSolid), 0.0) 
           -params.k5b * std::max((U5n1[i] * U8n[i] - std::pow(10., -params.pKCaCO3) / params.CSolid), 0.0);  
    else if (nodePhase[i] == 0)
      R5 = -params.k2b * std::max((U5n1[i] * U1n[i] - std::pow(10., -params.pKMgCO3) / params.CSolid), 0.0)
           -params.k3b * std::max((U5n1[i] * U1n[i] - std::pow(10., -params.pKMgHCO3) / params.CSolid), 0.0) 
           -params.k5b * std::max((U5n1[i] * U8n[i] - std::pow(10., -params.pKCaCO3) / params.CSolid), 0.0)
           +params.k8b * (std::pow(10., -params.pKHCO3) / params.CSolid - U2n[i] * U5n1[i]);  
    else
      R5 = 0.0;

    // Residual
    F5[i] = ((U5n1[i] - U5n[i]) / params.dt) + sum_KR51 + sum_KR52 - R5;
  }

// HPO4 2-
//#pragma omp declare reduction(merge : std::vector <Eigen::Triplet <double>>:
// omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for
  for (int i = 0; i < params.nodesNo; i++) {
    double R6 = 0.0;
    double sum_KR61 = 0.0;
    double sum_KR62 = 0.0;
    int interfaceFlag = 0;

    // Interface Nodes
    if (nodePhase[i] == 1)
      for (int j = 1; j < horizonsLengths[i]; j++) {
        int n = horizons(i, j);
        if (nodePhase[n] == 0) {
          interfaceFlag = 1;
          break;
        }
      }

    for (int j = 1; j < horizonsLengths[i]; j++) {
      int n = horizons(i, j);

      double xi_x = nodes(n, 0) - nodes(i, 0);
      double xi_y = nodes(n, 1) - nodes(i, 1);
      double xi_z = nodes(n, 2) - nodes(i, 2);

      double xi = std::sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);

      if (nodePhase[i] == 1 && nodePhase[n] == 1) {
        continue;
      } else if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        sum_KR61 += -(params.K6prop1 * params.D6Solid * (U6n1[n] - U6n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s61)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR62 += -(params.K6prop2 * params.D6Solid * params.z6 * params.F * U6n1[n] * (U11n[n] - U11n[i]) /
                      (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s62))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

      } else if (nodePhase[i] == 0 && nodePhase[n] == 1) {
        sum_KR61 += -(params.K6prop1 * params.D6Solid * (U6n1[n] - U6n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s61)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR62 += -(params.K6prop2 * params.D6Solid * params.z6 * params.F * U6n1[n] * (U11n[n] - U11n[i]) /
                      (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s62))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;
      } else {
        sum_KR61 += -(params.K6prop1 * params.D6Liquid * (U6n1[n] - U6n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s61)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR62 += -(params.K6prop2 * params.D6Liquid * params.z6 * params.F * U6n1[n] * (U11n[n] - U11n[i]) /
                    (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s62))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;
      }
    }

    if (nodePhase[i] == 1 && interfaceFlag == 1)
      R6 = -params.k4b * std::max((U1n[i] * U1n[i] * U1n[i] * U6n[i] * U6n[i] - std::pow(10., -params.pKMgPO4) / params.CSolid), 0.0);
    else if (nodePhase[i] == 0)
      R6 = -params.k4b * std::max((U1n[i] * U1n[i] * U1n[i] * U6n[i] * U6n[i] - std::pow(10., -params.pKMgPO4) / params.CSolid), 0.0)
           +params.k9b * (-std::pow(10., -params.pKHPO4) / params.CSolid + U7n[i] * U2n[i]);
    else
      R6 = 0.0;

    // Residual
    F6[i] = ((U6n1[i] - U6n[i]) / params.dt) + sum_KR61 + sum_KR62 - R6;
  }

// PO4 3-
//#pragma omp declare reduction(merge : std::vector <Eigen::Triplet <double>>:
// omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for
  for (int i = 0; i < params.nodesNo; i++) {
    double R7 = 0.0;
    double sum_KR71 = 0.0;
    double sum_KR72 = 0.0;
    int interfaceFlag = 0;

    // Interface Nodes
    if (nodePhase[i] == 1)
      for (int j = 1; j < horizonsLengths[i]; j++) {
        int n = horizons(i, j);
        if (nodePhase[n] == 0) {
          interfaceFlag = 1;
          break;
        }
      }

    for (int j = 1; j < horizonsLengths[i]; j++) {
      int n = horizons(i, j);

      double xi_x = nodes(n, 0) - nodes(i, 0);
      double xi_y = nodes(n, 1) - nodes(i, 1);
      double xi_z = nodes(n, 2) - nodes(i, 2);

      double xi = std::sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);

      if (nodePhase[i] == 1 && nodePhase[n] == 1) {
        continue;
      } else if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        sum_KR71 += -(params.K7prop1 * params.D7Solid * (U7n1[n] - U7n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s71)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR72 += -(params.K7prop2 * params.D7Solid * params.z7 * params.F * U7n1[n] * (U11n[n] - U11n[i]) /
                      (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s72))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

      } else if (nodePhase[i] == 0 && nodePhase[n] == 1) {
        sum_KR71 += -(params.K7prop1 * params.D7Solid * (U7n1[n] - U7n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s71)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR72 += -(params.K7prop2 * params.D7Solid * params.z7 * params.F * U7n1[n] * (U11n[n] - U11n[i]) /
                      (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s72))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;
      } else {
        sum_KR71 += -(params.K7prop1 * params.D7Liquid * (U7n1[n] - U7n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s71)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR72 += -(params.K7prop2 * params.D7Liquid * params.z7 * params.F * U7n1[n] * (U11n[n] - U11n[i]) /
                    (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s72))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;
      }
    }

    if (nodePhase[i] == 1 && interfaceFlag == 1)
      R7 = -params.k6b * std::max((U3n[i] * U8n[i] * U8n[i] * U8n[i] * U8n[i] * U8n[i] * U7n[i] * U7n[i] * U7n[i] - std::pow(10., -params.pKCa5OHPO4) / params.CSolid), 0.0);
    else if (nodePhase[i] == 0)
      R7 = -params.k6b * std::max((U3n[i] * U8n[i] * U8n[i] * U8n[i] * U8n[i] * U8n[i] * U7n[i] * U7n[i] * U7n[i] - std::pow(10., -params.pKCa5OHPO4) / params.CSolid), 0.0)
           +params.k9b * (std::pow(10., -params.pKHPO4) / params.CSolid - U7n1[i] * U2n[i]);
    else
      R7 = 0.0;

    // Residual
    F7[i] = ((U7n1[i] - U7n[i]) / params.dt) + sum_KR71 + sum_KR72 - R7;
  }

// Ca2+
//#pragma omp declare reduction(merge : std::vector <Eigen::Triplet <double>>:
// omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for
  for (int i = 0; i < params.nodesNo; i++) {
    double R8 = 0.0;
    double sum_KR81 = 0.0;
    double sum_KR82 = 0.0;
    int interfaceFlag = 0;

    // Interface Nodes
    if (nodePhase[i] == 1)
      for (int j = 1; j < horizonsLengths[i]; j++) {
        int n = horizons(i, j);
        if (nodePhase[n] == 0) {
          interfaceFlag = 1;
          break;
        }
      }

    for (int j = 1; j < horizonsLengths[i]; j++) {
      int n = horizons(i, j);

      double xi_x = nodes(n, 0) - nodes(i, 0);
      double xi_y = nodes(n, 1) - nodes(i, 1);
      double xi_z = nodes(n, 2) - nodes(i, 2);

      double xi = std::sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);

      if (nodePhase[i] == 1 && nodePhase[n] == 1) {
        continue;
      } else if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        sum_KR81 += -(params.K8prop1 * params.D8Solid * (U8n1[n] - U8n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s81)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR82 += -(params.K8prop2 * params.D8Solid * params.z8 * params.F * U8n1[n] * (U11n[n] - U11n[i]) /
                      (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s82))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

      } else if (nodePhase[i] == 0 && nodePhase[n] == 1) {
        sum_KR81 += -(params.K8prop1 * params.D8Solid * (U8n1[n] - U8n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s81)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR82 += -(params.K8prop2 * params.D8Solid * params.z8 * params.F * U8n1[n] * (U11n[n] - U11n[i]) /
                      (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s82))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;
      } else {
        sum_KR81 += -(params.K8prop1 * params.D8Liquid * (U8n1[n] - U8n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s81)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR82 += -(params.K8prop2 * params.D8Liquid * params.z8 * params.F * U8n1[n] * (U11n[n] - U11n[i]) /
                    (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s82))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;
      }
    }

    if (nodePhase[i] == 1 && interfaceFlag == 1)
      R8 = -params.k5b * std::max((U8n1[i] * U5n[i] - std::pow(10., -params.pKCaCO3) / params.CSolid), 0.0)
           -params.k6b * std::max((U3n[i] * U8n[i] * U8n[i] * U8n[i] * U8n[i] * U8n[i] * U7n[i] * U7n[i] * U7n[i] - std::pow(10., -params.pKCa5OHPO4) / params.CSolid), 0.0);
    else if (nodePhase[i] == 0)
      R8 = -params.k5b * std::max((U8n1[i] * U5n[i] - std::pow(10., -params.pKCaCO3) / params.CSolid), 0.0)
           -params.k6b * std::max((U3n[i] * U8n[i] * U8n[i] * U8n[i] * U8n[i] * U8n[i] * U7n[i] * U7n[i] * U7n[i] - std::pow(10., -params.pKCa5OHPO4) / params.CSolid), 0.0);
    else
      R8 = 0.0;

    // Residual
    F8[i] = ((U8n1[i] - U8n[i]) / params.dt) + sum_KR81 + sum_KR82 - R8;
  }

// Na+
//#pragma omp declare reduction(merge : std::vector <Eigen::Triplet <double>>:
// omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for
  for (int i = 0; i < params.nodesNo; i++) {
    double sum_KR91 = 0.0;
    double sum_KR92 = 0.0;

    for (int j = 1; j < horizonsLengths[i]; j++) {
      int n = horizons(i, j);

      double xi_x = nodes(n, 0) - nodes(i, 0);
      double xi_y = nodes(n, 1) - nodes(i, 1);
      double xi_z = nodes(n, 2) - nodes(i, 2);

      double xi = std::sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);

      if (nodePhase[i] == 1 && nodePhase[n] == 1) {
        continue;
      } else if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        sum_KR91 += -(params.K9prop1 * params.D9Solid * (U9n1[n] - U9n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s91)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR92 += -(params.K9prop2 * params.D9Solid * params.z9 * params.F * U9n1[n] * (U11n[n] - U11n[i]) /
                      (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s92))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

      } else if (nodePhase[i] == 0 && nodePhase[n] == 1) {
        sum_KR91 += -(params.K9prop1 * params.D9Solid * (U9n1[n] - U9n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s91)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR92 += -(params.K9prop2 * params.D9Solid * params.z9 * params.F * U9n1[n] * (U11n[n] - U11n[i]) /
                      (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s92))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;
      } else {
        sum_KR91 += -(params.K9prop1 * params.D9Liquid * (U9n1[n] - U9n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s91)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR92 += -(params.K9prop2 * params.D9Liquid * params.z9 * params.F * U9n1[n] * (U11n[n] - U11n[i]) /
                    (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s92))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;
      }
    }

    // Residual
    F9[i] = ((U9n1[i] - U9n[i]) / params.dt) + sum_KR91 + sum_KR92;
  }

// Cl-
//#pragma omp declare reduction(merge : std::vector <Eigen::Triplet <double>>:
// omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for
  for (int i = 0; i < params.nodesNo; i++) {
    double sum_KR101 = 0.0;
    double sum_KR102 = 0.0;

    for (int j = 1; j < horizonsLengths[i]; j++) {
      int n = horizons(i, j);

      double xi_x = nodes(n, 0) - nodes(i, 0);
      double xi_y = nodes(n, 1) - nodes(i, 1);
      double xi_z = nodes(n, 2) - nodes(i, 2);

      double xi = std::sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);

      if (nodePhase[i] == 1 && nodePhase[n] == 1) {
        continue;
      } else if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        sum_KR101 += -(params.K10prop1 * params.D10Solid * (U10n1[n] - U10n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s101)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR102 += -(params.K10prop2 * params.D10Solid * params.z10 * params.F * U10n1[n] * (U11n[n] - U11n[i]) /
                      (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s102))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

      } else if (nodePhase[i] == 0 && nodePhase[n] == 1) {
        sum_KR101 += -(params.K10prop1 * params.D10Solid * (U10n1[n] - U10n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s101)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR102 += -(params.K10prop2 * params.D10Solid * params.z10 * params.F * U10n1[n] * (U11n[n] - U11n[i]) /
                      (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s102))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;
      } else {
        sum_KR101 += -(params.K10prop1 * params.D10Liquid * (U10n1[n] - U10n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s101)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        sum_KR102 += -(params.K10prop2 * params.D10Liquid * params.z10 * params.F * U10n1[n] * (U11n[n] - U11n[i]) /
                    (params.R * params.T * std::pow(xi, 3.0 + 2.0 * params.s102))) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;
      }
    }

    // Residual
    F10[i] = ((U10n1[i] - U10n[i]) / params.dt) + sum_KR101 + sum_KR102;
  }

  // Phi
//#pragma omp declare reduction(merge : std::vector <Eigen::Triplet <double>>:
// omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for
  for (int i = 0; i < params.nodesNo; i++) {
    double R11 = 0.0;
    double sum_KR111 = 0.0;
    int Flag = 0;

    for (int j = 1; j < horizonsLengths[i]; j++) {
      int n = horizons(i, j);

      double xi_x = nodes(n, 0) - nodes(i, 0);
      double xi_y = nodes(n, 1) - nodes(i, 1);
      double xi_z = nodes(n, 2) - nodes(i, 2);

      double xi = std::sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);

      if (nodePhase[i] == 1 && nodePhase[n] == 1) {
        continue;
      } else if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        continue;
      } else if (nodePhase[i] == 0 && nodePhase[n] == 1) {
        continue;
      } else {
        sum_KR111 += -(params.K11prop1 * params.sigma1 * (U11n1[n] - U11n1[i]) /
                      std::pow(xi, 3.0 + 2.0 * params.s111)) *
                    utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        Flag = 1;
      }
    }

    if (Flag == 1) {
      R11 = params.z1 * U1n[i] + params.z2 * U2n[i] + params.z3 * U3n[i] + params.z4 * U4n[i] + params.z5 * U5n[i] +
            params.z6 * U6n[i] + params.z7 * U7n[i] + params.z8 * U8n[i] + params.z9 * U9n[i] + params.z10 * U10n[i];
    } else
      R11 = 0.0;

    // Residual
    if (Flag == 1)
      F11[i] = (1. / (params.K11prop1 * params.sigma1)) * sum_KR111 -
              (1. / (params.K11prop1 * params.sigma1)) * params.F * params.CSolid * R11;
  }
}

template <typename vector_Itype, typename vector_Dtype, typename matrix_Itype,
          typename matrix_Dtype, typename sparse_matrix_Dtype, typename parameters>
void NewtonMatrices(
    sparse_matrix_Dtype &sparseK1, sparse_matrix_Dtype &sparseK2,
    sparse_matrix_Dtype &sparseK3, sparse_matrix_Dtype &sparseK4,
    sparse_matrix_Dtype &sparseK5, sparse_matrix_Dtype &sparseK6,
    sparse_matrix_Dtype &sparseK7, sparse_matrix_Dtype &sparseK8,
    sparse_matrix_Dtype &sparseK9, sparse_matrix_Dtype &sparseK10,
    sparse_matrix_Dtype &sparseK11,
    matrix_Dtype &nodes, matrix_Itype &horizons,
    vector_Dtype &boundaryConditionValues1,
    vector_Dtype &boundaryConditionValues2,
    vector_Dtype &boundaryConditionValues3,
    vector_Dtype &boundaryConditionValues4,
    vector_Dtype &boundaryConditionValues5, 
    vector_Dtype &boundaryConditionValues6,
    vector_Dtype &boundaryConditionValues7,
    vector_Dtype &boundaryConditionValues8,
    vector_Dtype &boundaryConditionValues9,
    vector_Dtype &boundaryConditionValues10,
    vector_Dtype &boundaryConditionValues11,
    vector_Dtype &U1n1, vector_Dtype &U2n1, vector_Dtype &U3n1, vector_Dtype &U4n1,
    vector_Dtype &U5n1, vector_Dtype &U6n1, vector_Dtype &U7n1, vector_Dtype &U8n1, 
    vector_Dtype &U9n1, vector_Dtype &U10n1,vector_Dtype &U11n1,
    vector_Dtype &U1n, vector_Dtype &U2n, vector_Dtype &U3n, vector_Dtype &U4n,
    vector_Dtype &U5n, vector_Dtype &U6n, vector_Dtype &U7n, vector_Dtype &U8n, 
    vector_Dtype &U9n, vector_Dtype &U10n,vector_Dtype &U11n,
    vector_Itype &nodePhase, vector_Itype &horizonsLengths, vector_Itype &boundaryConditionTypes,
    double voln1rel, parameters &params) {
  std::vector<Eigen::Triplet<double>> coefficientsSparseK1;
  std::vector<Eigen::Triplet<double>> coefficientsSparseK2;
  std::vector<Eigen::Triplet<double>> coefficientsSparseK3;
  std::vector<Eigen::Triplet<double>> coefficientsSparseK4;
  std::vector<Eigen::Triplet<double>> coefficientsSparseK5;
  std::vector<Eigen::Triplet<double>> coefficientsSparseK6;
  std::vector<Eigen::Triplet<double>> coefficientsSparseK7;
  std::vector<Eigen::Triplet<double>> coefficientsSparseK8;
  std::vector<Eigen::Triplet<double>> coefficientsSparseK9;
  std::vector<Eigen::Triplet<double>> coefficientsSparseK10;
  std::vector<Eigen::Triplet<double>> coefficientsSparseK11;

  sparseK1.setZero();
  sparseK2.setZero();
  sparseK3.setZero();
  sparseK4.setZero();
  sparseK5.setZero();
  sparseK6.setZero();
  sparseK7.setZero();
  sparseK8.setZero();
  sparseK9.setZero();
  sparseK10.setZero();
  sparseK11.setZero();

// Mg
#pragma omp declare reduction(merge : std::vector <Eigen::Triplet <double>>: omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge : coefficientsSparseK1)
  for (int i = 0; i < params.nodesNo; i++) {
    double R1 = 0.0;
    double sum_K1 = 0.0;
    double phi_c = 0.0;
    double BV = 1.0;
    double eta = 0.0;
    int interfaceFlag = 0;

    if (boundaryConditionTypes[i] == 0 && boundaryConditionValues1[i] != -1) {
      coefficientsSparseK1.push_back(Eigen::Triplet<double>(i, i, 1.0));
      continue;
    }

    // Interface Nodes
    if (nodePhase[i] == 1)
      for (int j = 1; j < horizonsLengths[i]; j++) {
        int n = horizons(i, j);
        if (nodePhase[n] == 0) {
          interfaceFlag = 1;
          break;
        }
      }

    for (int j = 1; j < horizonsLengths[i]; j++) {
      int n = horizons(i, j);

      double xi_x = nodes(n, 0) - nodes(i, 0);
      double xi_y = nodes(n, 1) - nodes(i, 1);
      double xi_z = nodes(n, 2) - nodes(i, 2);

      double xi = std::sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);

      if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        /*phi_c = (params.R * params.T) / (params.F * params.z1) * std::log(U1n[i] / (params.CSat / params.CSolid));

        eta = params.phim + params.phimse - U11n[n] - phi_c;

        BV = std::exp(params.alphaA * params.z1 * params.F * eta / (params.R * params.T));*/
        BV = std::pow(10., -params.vol_red * (1. - voln1rel));
      } else if (nodePhase[n] == 1 && nodePhase[i] == 0) {
        /* phi_c = (params.R * params.T) / (params.F * params.z1) * std::log(U1n[n] / (params.CSat / params.CSolid));

        eta = params.phim + params.phimse - U11n[i] - phi_c;

        BV = std::exp(params.alphaA * params.z1 * params.F * eta / (params.R * params.T));*/
        BV = std::pow(10., -params.vol_red * (1. - voln1rel));
      } else {
        BV = 1.0;
      }

      // BV = 1.0;

      if (nodePhase[i] == 1 && nodePhase[n] == 1) {
        // continue;
        // Tangent stiffness
        sum_K1 += -(params.K1prop1 * params.D1Solid * BV / std::pow(xi, 3.0 + 2.0 * params.s11)) *
                 utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;
        
        coefficientsSparseK1.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K1prop1 * params.D1Solid * BV / std::pow(xi, 3.0 + 2.0 * params.s11)) +
              (params.K1prop2 * params.D1Solid * BV * params.z1 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
              std::pow(xi, 3.0 + 2.0 * params.s12))) *
              utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));
      } else if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        // Tangent stiffness
        sum_K1 += -(params.K1prop1 * params.D1Solid * BV / std::pow(xi, 3.0 + 2.0 * params.s11)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK1.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K1prop1 * params.D1Solid * BV / std::pow(xi, 3.0 + 2.0 * params.s11)) +
              (params.K1prop2 * params.D1Solid * BV * params.z1 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s12))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));

      } else if (nodePhase[i] == 0 && nodePhase[n] == 1) {
        // Tangent stiffness
        sum_K1 += -(params.K1prop1 * params.D1Solid * BV / std::pow(xi, 3.0 + 2.0 * params.s11)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK1.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K1prop1 * params.D1Solid * BV / std::pow(xi, 3.0 + 2.0 * params.s11)) +
              (params.K1prop2 * params.D1Solid * BV * params.z1 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s12))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));
      } else {
        // Tangent stiffness
        sum_K1 += -(params.K1prop1 * params.D1Liquid / std::pow(xi, 3.0 + 2.0 * params.s11)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK1.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K1prop1 * params.D1Liquid / std::pow(xi, 3.0 + 2.0 * params.s11)) +
              (params.K1prop2 * params.D1Liquid * params.z1 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s12))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));
      }
    }

    if (/*nodePhase[i] == 1 &&*/ interfaceFlag == 1) 
      R1 = 0.0;
    else if (nodePhase[i] == 0)
      R1 = 0.0;
    else
      R1 = 0.0;

    // Tangent stiffness
    coefficientsSparseK1.push_back(
        Eigen::Triplet<double>(i, i, (1. / params.dt) - sum_K1 - R1));
  }

  // Fill matrices from triplets
  sparseK1.setFromTriplets(coefficientsSparseK1.begin(),
                           coefficientsSparseK1.end());

// H+
//#pragma omp declare reduction(merge : std::vector <Eigen::Triplet <double>>:
// omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge : coefficientsSparseK2)
  for (int i = 0; i < params.nodesNo; i++) {
    double R2 = 0.0;
    double sum_K2 = 0.0;
    int interfaceFlag = 0;

    if (boundaryConditionTypes[i] == 0 && boundaryConditionValues2[i] != -1) {
      coefficientsSparseK2.push_back(Eigen::Triplet<double>(i, i, 1.0));
      continue;
    }

    // Interface Nodes
    if (nodePhase[i] == 1)
      for (int j = 1; j < horizonsLengths[i]; j++) {
        int n = horizons(i, j);
        if (nodePhase[n] == 0) {
          interfaceFlag = 1;
          break;
        }
      }

    for (int j = 1; j < horizonsLengths[i]; j++) {
      int n = horizons(i, j);

      double xi_x = nodes(n, 0) - nodes(i, 0);
      double xi_y = nodes(n, 1) - nodes(i, 1);
      double xi_z = nodes(n, 2) - nodes(i, 2);

      double xi = std::sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);

      if (nodePhase[i] == 1 && nodePhase[n] == 1) {
        continue;
      } else if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        // Tangent stiffness
        sum_K2 += -(params.K2prop1 * params.D2Solid / std::pow(xi, 3.0 + 2.0 * params.s21)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK2.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K2prop1 * params.D2Solid / std::pow(xi, 3.0 + 2.0 * params.s21)) +
              (params.K2prop2 * params.D2Solid * params.z2 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s22))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));

      } else if (nodePhase[i] == 0 && nodePhase[n] == 1) {
        // Tangent stiffness
        sum_K2 += -(params.K2prop1 * params.D2Solid / std::pow(xi, 3.0 + 2.0 * params.s21)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK2.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K2prop1 * params.D2Solid / std::pow(xi, 3.0 + 2.0 * params.s21)) +
              (params.K2prop2 * params.D2Solid * params.z2 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s22))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));
      } else {
        // Tangent stiffness
        sum_K2 += -(params.K2prop1 * params.D2Liquid / std::pow(xi, 3.0 + 2.0 * params.s21)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK2.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K2prop1 * params.D2Liquid / std::pow(xi, 3.0 + 2.0 * params.s21)) +
              (params.K2prop2 * params.D2Liquid * params.z2 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s22))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));
      }
    }

    if (/*nodePhase[i] == 1 &&*/ interfaceFlag == 1) {
      R2 =  params.k7b * (- U3n[i])
          + params.k8b * (- U5n[i])
          + params.k9b * (- U7n[i]); 
    } 
    else if (nodePhase[i] == 0){
      R2 =  params.k7b * (- U3n[i])
          + params.k8b * (- U5n[i])
          + params.k9b * (- U7n[i]);     
    }
    else
      R2 = 0.0;

    // Tangent stiffness
    coefficientsSparseK2.push_back(
        Eigen::Triplet<double>(i, i, (1. / params.dt) - sum_K2 - R2));
  }

  // Fill matrices from triplets
  sparseK2.setFromTriplets(coefficientsSparseK2.begin(),
                           coefficientsSparseK2.end());

  // OH-
//#pragma omp declare reduction(merge : std::vector <Eigen::Triplet <double>>:
// omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge : coefficientsSparseK3)
  for (int i = 0; i < params.nodesNo; i++) {
    double R3 = 0.0;
    double sum_K3 = 0.0;
    int interfaceFlag = 0;

    if (boundaryConditionTypes[i] == 0 && boundaryConditionValues3[i] != -1) {
      coefficientsSparseK3.push_back(Eigen::Triplet<double>(i, i, 1.0));
      continue;
    }

    // Interface Nodes
    if (nodePhase[i] == 1)
      for (int j = 1; j < horizonsLengths[i]; j++) {
        int n = horizons(i, j);
        if (nodePhase[n] == 0) {
          interfaceFlag = 1;
          break;
        }
      }

    for (int j = 1; j < horizonsLengths[i]; j++) {
      int n = horizons(i, j);

      double xi_x = nodes(n, 0) - nodes(i, 0);
      double xi_y = nodes(n, 1) - nodes(i, 1);
      double xi_z = nodes(n, 2) - nodes(i, 2);

      double xi = std::sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);

      if (nodePhase[i] == 1 && nodePhase[n] == 1) {
        continue;
      } else if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        // Tangent stiffness
        sum_K3 += -(params.K3prop1 * params.D3Solid / std::pow(xi, 3.0 + 2.0 * params.s31)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK3.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K3prop1 * params.D3Solid / std::pow(xi, 3.0 + 2.0 * params.s31)) +
              (params.K3prop2 * params.D3Solid * params.z3 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s32))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));

      } else if (nodePhase[i] == 0 && nodePhase[n] == 1) {
        // Tangent stiffness
        sum_K3 += -(params.K3prop1 * params.D3Solid / std::pow(xi, 3.0 + 2.0 * params.s31)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK3.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K3prop1 * params.D3Solid / std::pow(xi, 3.0 + 2.0 * params.s31)) +
              (params.K3prop2 * params.D3Solid * params.z3 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s32))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));

      } else {
        // Tangent stiffness
        sum_K3 += -(params.K3prop1 * params.D3Liquid / std::pow(xi, 3.0 + 2.0 * params.s31)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK3.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K3prop1 * params.D3Liquid / std::pow(xi, 3.0 + 2.0 * params.s31)) +
              (params.K3prop2 * params.D3Liquid * params.z3 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s32))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));
      }
    }

    if (/*nodePhase[i] == 1 &&*/ interfaceFlag == 1) {
      R3 = 0.0;
    }
    else if (nodePhase[i] == 0) {
      R3 = params.k7b * (-U2n[i]);
    } else
      R3 = 0.0;

    // Tangent stiffness
    coefficientsSparseK3.push_back(
        Eigen::Triplet<double>(i, i, (1. / params.dt) - sum_K3 - R3));
  }

  // Fill matrices from triplets
  sparseK3.setFromTriplets(coefficientsSparseK3.begin(),
                           coefficientsSparseK3.end());

// HCO3 -
//#pragma omp declare reduction(merge : std::vector <Eigen::Triplet <double>>:
// omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge : coefficientsSparseK4)
  for (int i = 0; i < params.nodesNo; i++) {
    double R4 = 0.0;
    double sum_K4 = 0.0;
    int interfaceFlag = 0;

    if (boundaryConditionTypes[i] == 0 && boundaryConditionValues4[i] != -1) {
      coefficientsSparseK4.push_back(Eigen::Triplet<double>(i, i, 1.0));
      continue;
    }

    // Interface Nodes
    if (nodePhase[i] == 1)
      for (int j = 1; j < horizonsLengths[i]; j++) {
        int n = horizons(i, j);
        if (nodePhase[n] == 0) {
          interfaceFlag = 1;
          break;
        }
      }

    for (int j = 1; j < horizonsLengths[i]; j++) {
      int n = horizons(i, j);

      double xi_x = nodes(n, 0) - nodes(i, 0);
      double xi_y = nodes(n, 1) - nodes(i, 1);
      double xi_z = nodes(n, 2) - nodes(i, 2);

      double xi = std::sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);

      if (nodePhase[i] == 1 && nodePhase[n] == 1) {
        continue;
      } else if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        // Tangent stiffness
        sum_K4 += -(params.K4prop1 * params.D4Solid / std::pow(xi, 3.0 + 2.0 * params.s41)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK4.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K4prop1 * params.D4Solid / std::pow(xi, 3.0 + 2.0 * params.s41)) +
              (params.K4prop2 * params.D4Solid * params.z4 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s42))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));

      } else if (nodePhase[i] == 0 && nodePhase[n] == 1) {
        // Tangent stiffness
        sum_K4 += -(params.K4prop1 * params.D4Solid / std::pow(xi, 3.0 + 2.0 * params.s41)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK4.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K4prop1 * params.D4Solid / std::pow(xi, 3.0 + 2.0 * params.s41)) +
              (params.K4prop2 * params.D4Solid * params.z4 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s42))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));

      } else {
        // Tangent stiffness
        sum_K4 += -(params.K4prop1 * params.D4Liquid / std::pow(xi, 3.0 + 2.0 * params.s41)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK4.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K4prop1 * params.D4Liquid / std::pow(xi, 3.0 + 2.0 * params.s41)) +
              (params.K4prop2 * params.D4Liquid * params.z4 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s42))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));
      }
    }

    if (nodePhase[i] == 1 && interfaceFlag == 1)
      R4 = 0.0;
    else if (nodePhase[i] == 0)
      R4 = params.k8b * (- U2n[i]);
    else
      R4 = 0.0;

    // Tangent stiffness
    coefficientsSparseK4.push_back(
        Eigen::Triplet<double>(i, i, (1. / params.dt) - sum_K4 - R4));
  }

  // Fill matrices from triplets
  sparseK4.setFromTriplets(coefficientsSparseK4.begin(),
                           coefficientsSparseK4.end());

  
// CO3 2-
//#pragma omp declare reduction(merge : std::vector <Eigen::Triplet <double>>:
// omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge : coefficientsSparseK5)
  for (int i = 0; i < params.nodesNo; i++) {
    double R5 = 0.0;
    double sum_K5 = 0.0;
    int interfaceFlag = 0;

    if (boundaryConditionTypes[i] == 0 && boundaryConditionValues5[i] != -1) {
      coefficientsSparseK5.push_back(Eigen::Triplet<double>(i, i, 1.0));
      continue;
    }

    // Interface Nodes
    if (nodePhase[i] == 1)
      for (int j = 1; j < horizonsLengths[i]; j++) {
        int n = horizons(i, j);
        if (nodePhase[n] == 0) {
          interfaceFlag = 1;
          break;
        }
      }

    for (int j = 1; j < horizonsLengths[i]; j++) {
      int n = horizons(i, j);

      double xi_x = nodes(n, 0) - nodes(i, 0);
      double xi_y = nodes(n, 1) - nodes(i, 1);
      double xi_z = nodes(n, 2) - nodes(i, 2);

      double xi = std::sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);

      if (nodePhase[i] == 1 && nodePhase[n] == 1) {
        continue;
      } else if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        // Tangent stiffness
        sum_K5 += -(params.K5prop1 * params.D5Solid / std::pow(xi, 3.0 + 2.0 * params.s51)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK5.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K5prop1 * params.D5Solid / std::pow(xi, 3.0 + 2.0 * params.s51)) +
              (params.K5prop2 * params.D5Solid * params.z5 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s52))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));

      } else if (nodePhase[i] == 0 && nodePhase[n] == 1) {
        // Tangent stiffness
        sum_K5 += -(params.K5prop1 * params.D5Solid / std::pow(xi, 3.0 + 2.0 * params.s51)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK5.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K5prop1 * params.D5Solid / std::pow(xi, 3.0 + 2.0 * params.s51)) +
              (params.K5prop2 * params.D5Solid * params.z5 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s52))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));

      } else {
        // Tangent stiffness
        sum_K5 += -(params.K5prop1 * params.D5Liquid / std::pow(xi, 3.0 + 2.0 * params.s51)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK5.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K5prop1 * params.D5Liquid / std::pow(xi, 3.0 + 2.0 * params.s51)) +
              (params.K5prop2 * params.D5Liquid * params.z5 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s52))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));
      }
    }

    if (nodePhase[i] == 1 && interfaceFlag == 1)
      R5 = 0.0; 
    else if (nodePhase[i] == 0)
      R5 = params.k8b * (- U2n[i]);  
    else
      R5 = 0.0;

    // Tangent stiffness
    coefficientsSparseK5.push_back(
        Eigen::Triplet<double>(i, i, (1. / params.dt) - sum_K5 - R5));
  }

  // Fill matrices from triplets
  sparseK5.setFromTriplets(coefficientsSparseK5.begin(),
                           coefficientsSparseK5.end());


// HPO4 2-
//#pragma omp declare reduction(merge : std::vector <Eigen::Triplet <double>>:
// omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge : coefficientsSparseK6)
  for (int i = 0; i < params.nodesNo; i++) {
    double R6 = 0.0;
    double sum_K6 = 0.0;
    int interfaceFlag = 0;

    if (boundaryConditionTypes[i] == 0 && boundaryConditionValues6[i] != -1) {
      coefficientsSparseK6.push_back(Eigen::Triplet<double>(i, i, 1.0));
      continue;
    }

    // Interface Nodes
    if (nodePhase[i] == 1)
      for (int j = 1; j < horizonsLengths[i]; j++) {
        int n = horizons(i, j);
        if (nodePhase[n] == 0) {
          interfaceFlag = 1;
          break;
        }
      }

    for (int j = 1; j < horizonsLengths[i]; j++) {
      int n = horizons(i, j);

      double xi_x = nodes(n, 0) - nodes(i, 0);
      double xi_y = nodes(n, 1) - nodes(i, 1);
      double xi_z = nodes(n, 2) - nodes(i, 2);

      double xi = std::sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);

      if (nodePhase[i] == 1 && nodePhase[n] == 1) {
        continue;
      } else if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        // Tangent stiffness
        sum_K6 += -(params.K6prop1 * params.D6Solid / std::pow(xi, 3.0 + 2.0 * params.s61)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK6.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K6prop1 * params.D6Solid / std::pow(xi, 3.0 + 2.0 * params.s61)) +
              (params.K6prop2 * params.D6Solid * params.z6 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s62))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));

      } else if (nodePhase[i] == 0 && nodePhase[n] == 1) {
        // Tangent stiffness
        sum_K6 += -(params.K6prop1 * params.D6Solid / std::pow(xi, 3.0 + 2.0 * params.s61)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK6.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K6prop1 * params.D6Solid / std::pow(xi, 3.0 + 2.0 * params.s61)) +
              (params.K6prop2 * params.D6Solid * params.z6 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s62))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));

      } else {
        // Tangent stiffness
        sum_K6 += -(params.K6prop1 * params.D6Liquid / std::pow(xi, 3.0 + 2.0 * params.s61)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK6.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K6prop1 * params.D6Liquid / std::pow(xi, 3.0 + 2.0 * params.s61)) +
              (params.K6prop2 * params.D6Liquid * params.z6 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s62))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));
      }
    }

    if (nodePhase[i] == 1 && interfaceFlag == 1)
      R6 = 0.0;
    else if (nodePhase[i] == 0)
      R6 = 0.0;
    else
      R6 = 0.0;


    // Tangent stiffness
    coefficientsSparseK6.push_back(
        Eigen::Triplet<double>(i, i, (1. / params.dt) - sum_K6 - R6));
  }

  // Fill matrices from triplets
  sparseK6.setFromTriplets(coefficientsSparseK6.begin(),
                           coefficientsSparseK6.end());


// PO4 3-
//#pragma omp declare reduction(merge : std::vector <Eigen::Triplet <double>>:
// omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge : coefficientsSparseK7)
  for (int i = 0; i < params.nodesNo; i++) {
    double R7 = 0.0;
    double sum_K7 = 0.0;
    int interfaceFlag = 0;

    if (boundaryConditionTypes[i] == 0 && boundaryConditionValues7[i] != -1) {
      coefficientsSparseK7.push_back(Eigen::Triplet<double>(i, i, 1.0));
      continue;
    }

    // Interface Nodes
    if (nodePhase[i] == 1)
      for (int j = 1; j < horizonsLengths[i]; j++) {
        int n = horizons(i, j);
        if (nodePhase[n] == 0) {
          interfaceFlag = 1;
          break;
        }
      }

    for (int j = 1; j < horizonsLengths[i]; j++) {
      int n = horizons(i, j);

      double xi_x = nodes(n, 0) - nodes(i, 0);
      double xi_y = nodes(n, 1) - nodes(i, 1);
      double xi_z = nodes(n, 2) - nodes(i, 2);

      double xi = std::sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);

      if (nodePhase[i] == 1 && nodePhase[n] == 1) {
        continue;
      } else if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        // Tangent stiffness
        sum_K7 += -(params.K7prop1 * params.D7Solid / std::pow(xi, 3.0 + 2.0 * params.s71)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK7.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K7prop1 * params.D7Solid / std::pow(xi, 3.0 + 2.0 * params.s71)) +
              (params.K7prop2 * params.D7Solid * params.z7 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s72))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));

      } else if (nodePhase[i] == 0 && nodePhase[n] == 1) {
        // Tangent stiffness
        sum_K7 += -(params.K7prop1 * params.D7Solid / std::pow(xi, 3.0 + 2.0 * params.s71)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK7.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K7prop1 * params.D7Solid / std::pow(xi, 3.0 + 2.0 * params.s71)) +
              (params.K7prop2 * params.D7Solid * params.z7 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s72))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));

      } else {
        // Tangent stiffness
        sum_K7 += -(params.K7prop1 * params.D7Liquid / std::pow(xi, 3.0 + 2.0 * params.s71)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK7.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K7prop1 * params.D7Liquid / std::pow(xi, 3.0 + 2.0 * params.s71)) +
              (params.K7prop2 * params.D7Liquid * params.z7 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s72))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));
      }
    }

    if (nodePhase[i] == 1 && interfaceFlag == 1)
      R7 = 0.0;
    else if (nodePhase[i] == 0)
      R7 = params.k9b * (- U2n[i]);
    else
      R7 = 0.0;

    // Tangent stiffness
    coefficientsSparseK7.push_back(
        Eigen::Triplet<double>(i, i, (1. / params.dt) - sum_K7 - R7));
  }

  // Fill matrices from triplets
  sparseK7.setFromTriplets(coefficientsSparseK7.begin(),
                           coefficientsSparseK7.end());


// Ca2+
//#pragma omp declare reduction(merge : std::vector <Eigen::Triplet <double>>:
// omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge : coefficientsSparseK8)
  for (int i = 0; i < params.nodesNo; i++) {
    double R8 = 0.0;
    double sum_K8 = 0.0;
    int interfaceFlag = 0;

    if (boundaryConditionTypes[i] == 0 && boundaryConditionValues8[i] != -1) {
      coefficientsSparseK8.push_back(Eigen::Triplet<double>(i, i, 1.0));
      continue;
    }

    // Interface Nodes
    if (nodePhase[i] == 1)
      for (int j = 1; j < horizonsLengths[i]; j++) {
        int n = horizons(i, j);
        if (nodePhase[n] == 0) {
          interfaceFlag = 1;
          break;
        }
      }

    for (int j = 1; j < horizonsLengths[i]; j++) {
      int n = horizons(i, j);

      double xi_x = nodes(n, 0) - nodes(i, 0);
      double xi_y = nodes(n, 1) - nodes(i, 1);
      double xi_z = nodes(n, 2) - nodes(i, 2);

      double xi = std::sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);

      if (nodePhase[i] == 1 && nodePhase[n] == 1) {
        continue;
      } else if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        // Tangent stiffness
        sum_K8 += -(params.K8prop1 * params.D8Solid / std::pow(xi, 3.0 + 2.0 * params.s81)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK8.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K8prop1 * params.D8Solid / std::pow(xi, 3.0 + 2.0 * params.s81)) +
              (params.K8prop2 * params.D8Solid * params.z8 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s82))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));

      } else if (nodePhase[i] == 0 && nodePhase[n] == 1) {
        // Tangent stiffness
        sum_K8 += -(params.K8prop1 * params.D8Solid / std::pow(xi, 3.0 + 2.0 * params.s81)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK8.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K8prop1 * params.D8Solid / std::pow(xi, 3.0 + 2.0 * params.s81)) +
              (params.K8prop2 * params.D8Solid * params.z8 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s82))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));

      } else {
        // Tangent stiffness
        sum_K8 += -(params.K8prop1 * params.D8Liquid / std::pow(xi, 3.0 + 2.0 * params.s81)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK8.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K8prop1 * params.D8Liquid / std::pow(xi, 3.0 + 2.0 * params.s81)) +
              (params.K8prop2 * params.D8Liquid * params.z8 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s82))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));
      }
    }

    if (nodePhase[i] == 1 && interfaceFlag == 1)
      R8 = 0.0;
    else if (nodePhase[i] == 0)
      R8 = 0.0;
    else
      R8 = 0.0;

    // Tangent stiffness
    coefficientsSparseK8.push_back(
        Eigen::Triplet<double>(i, i, (1. / params.dt) - sum_K8 - R8));
  }

  // Fill matrices from triplets
  sparseK8.setFromTriplets(coefficientsSparseK8.begin(),
                           coefficientsSparseK8.end());


// Na+
//#pragma omp declare reduction(merge : std::vector <Eigen::Triplet <double>>:
// omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge : coefficientsSparseK9)
  for (int i = 0; i < params.nodesNo; i++) {
    double sum_K9 = 0.0;

    if (boundaryConditionTypes[i] == 0 && boundaryConditionValues9[i] != -1) {
      coefficientsSparseK9.push_back(Eigen::Triplet<double>(i, i, 1.0));
      continue;
    }

    for (int j = 1; j < horizonsLengths[i]; j++) {
      int n = horizons(i, j);

      double xi_x = nodes(n, 0) - nodes(i, 0);
      double xi_y = nodes(n, 1) - nodes(i, 1);
      double xi_z = nodes(n, 2) - nodes(i, 2);

      double xi = std::sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);

      if (nodePhase[i] == 1 && nodePhase[n] == 1) {
        continue;
      } else if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        // Tangent stiffness
        sum_K9 += -(params.K9prop1 * params.D9Solid / std::pow(xi, 3.0 + 2.0 * params.s91)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK9.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K9prop1 * params.D9Solid / std::pow(xi, 3.0 + 2.0 * params.s91)) +
              (params.K9prop2 * params.D9Solid * params.z9 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s92))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));

      } else if (nodePhase[i] == 0 && nodePhase[n] == 1) {
        // Tangent stiffness
        sum_K9 += -(params.K9prop1 * params.D9Solid / std::pow(xi, 3.0 + 2.0 * params.s91)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK9.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K9prop1 * params.D9Solid / std::pow(xi, 3.0 + 2.0 * params.s91)) +
              (params.K9prop2 * params.D9Solid * params.z9 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s92))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));

      } else {
        // Tangent stiffness
        sum_K9 += -(params.K9prop1 * params.D9Liquid / std::pow(xi, 3.0 + 2.0 * params.s91)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK9.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K9prop1 * params.D9Liquid / std::pow(xi, 3.0 + 2.0 * params.s91)) +
              (params.K9prop2 * params.D9Liquid * params.z9 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s92))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));
      }
    }

    // Tangent stiffness
    coefficientsSparseK9.push_back(
        Eigen::Triplet<double>(i, i, (1. / params.dt) - sum_K9));
  }

  // Fill matrices from triplets
  sparseK9.setFromTriplets(coefficientsSparseK9.begin(),
                           coefficientsSparseK9.end());

// Cl-
//#pragma omp declare reduction(merge : std::vector <Eigen::Triplet <double>>:
// omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge : coefficientsSparseK10)
  for (int i = 0; i < params.nodesNo; i++) {
    double sum_K10 = 0.0;

    if (boundaryConditionTypes[i] == 0 && boundaryConditionValues10[i] != -1) {
      coefficientsSparseK10.push_back(Eigen::Triplet<double>(i, i, 1.0));
      continue;
    }

    for (int j = 1; j < horizonsLengths[i]; j++) {
      int n = horizons(i, j);

      double xi_x = nodes(n, 0) - nodes(i, 0);
      double xi_y = nodes(n, 1) - nodes(i, 1);
      double xi_z = nodes(n, 2) - nodes(i, 2);

      double xi = std::sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);

      if (nodePhase[i] == 1 && nodePhase[n] == 1) {
        continue;
      } else if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        // Tangent stiffness
        sum_K10 += -(params.K10prop1 * params.D10Solid / std::pow(xi, 3.0 + 2.0 * params.s101)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK10.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K10prop1 * params.D10Solid / std::pow(xi, 3.0 + 2.0 * params.s101)) +
              (params.K10prop2 * params.D10Solid * params.z10 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s102))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));

      } else if (nodePhase[i] == 0 && nodePhase[n] == 1) {
        // Tangent stiffness
        sum_K10 += -(params.K10prop1 * params.D10Solid / std::pow(xi, 3.0 + 2.0 * params.s101)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK10.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K10prop1 * params.D10Solid / std::pow(xi, 3.0 + 2.0 * params.s101)) +
              (params.K10prop2 * params.D10Solid * params.z10 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s102))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));

      } else {
        // Tangent stiffness
        sum_K10 += -(params.K10prop1 * params.D10Liquid / std::pow(xi, 3.0 + 2.0 * params.s101)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK10.push_back(Eigen::Triplet<double>(
            i, n,
            -((params.K10prop1 * params.D10Liquid / std::pow(xi, 3.0 + 2.0 * params.s101)) +
              (params.K10prop2 * params.D10Liquid * params.z10 * params.F / (params.R * params.T) * (U11n[n] - U11n[i]) /
               std::pow(xi, 3.0 + 2.0 * params.s102))) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));
      }
    }

    // Tangent stiffness
    coefficientsSparseK10.push_back(
        Eigen::Triplet<double>(i, i, (1. / params.dt) - sum_K10));
  }

  // Fill matrices from triplets
  sparseK10.setFromTriplets(coefficientsSparseK10.begin(),
                            coefficientsSparseK10.end());



// Phi
//#pragma omp declare reduction(merge : std::vector <Eigen::Triplet <double>>:
// omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for reduction(merge : coefficientsSparseK11)
  for (int i = 0; i < params.nodesNo; i++) {
    double sum_K11 = 0.0;
    int Flag = 0;

    if (boundaryConditionTypes[i] == 0 && boundaryConditionValues11[i] != -1) {
      coefficientsSparseK11.push_back(Eigen::Triplet<double>(i, i, 1.0));
      continue;
    }

    for (int j = 1; j < horizonsLengths[i]; j++) {
      int n = horizons(i, j);

      double xi_x = nodes(n, 0) - nodes(i, 0);
      double xi_y = nodes(n, 1) - nodes(i, 1);
      double xi_z = nodes(n, 2) - nodes(i, 2);

      double xi = std::sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);

      if (nodePhase[i] == 1 && nodePhase[n] == 1) {
        continue;
      } else if (nodePhase[i] == 1 && nodePhase[n] == 0) {
        continue;
      } else if (nodePhase[i] == 0 && nodePhase[n] == 1) {
        continue;
      } else {
        // Tangent stiffness
        sum_K11 += -(params.K11prop1 * params.sigma1 / std::pow(xi, 3.0 + 2.0 * params.s111)) *
                  utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz;

        coefficientsSparseK11.push_back(Eigen::Triplet<double>(
            i, n,
            -(1. / (params.K11prop1 * params.sigma1)) *
                (params.K11prop1 * params.sigma1 / std::pow(xi, 3.0 + 2.0 * params.s111)) *
                utils::Beta(xi, params.delta, params.dx) * params.dx * params.dy * params.dz));

        Flag = 1;
      }
    }

    // Tangent stiffness
    if (Flag == 1)
      coefficientsSparseK11.push_back(
          Eigen::Triplet<double>(i, i,
                                  -(1. / (params.K11prop1 * params.sigma1)) * sum_K11));
    else
      coefficientsSparseK11.push_back(Eigen::Triplet<double>(i, i, 1.0));
  }

  // Fill matrices from triplets
  sparseK11.setFromTriplets(coefficientsSparseK11.begin(),
                            coefficientsSparseK11.end());
}

template <typename vector_type>
bool read_vector_from_file(const std::string &filename, vector_type &vec,
                           int size) {
  std::ifstream file(filename.c_str());

  if (!file) {
    return false;
  }

  vec.resize(size);

  for (int i = 0; i < size; i++) {
    file >> vec[i];
  }

  return true;
}

template <typename matrix_type>
bool read_matrix_from_file(const std::string &filename, matrix_type &mat,
                           int size1, int size2) {
  std::ifstream file(filename.c_str());

  if (!file) {
    return false;
  }

  mat.resize(size1, size2);

  for (int i = 0; i < size1; i++) {
    for (int j = 0; j < size2; j++) {
      file >> mat(i, j);
    }
  }

  return true;
}

template <typename vector_type>
bool write_vector_to_file(const std::string &filename, const vector_type &vec) {
  std::ofstream file(filename.c_str());

  if (!file) {
    return false;
  }

  file.precision(16);

  for (int i = 0; i < vec.size(); i++) {
    file << vec[i] << std::endl;
  }

  return true;
}

template <typename matrix_type>
bool write_matrix_to_file(const std::string &filename, const matrix_type &mat) {
  std::ofstream file(filename.c_str());

  if (!file) {
    return false;
  }

  file.precision(16);

  for (int i = 0; i < mat.rows(); i++) {
    for (int j = 0; j < mat.cols(); j++) {
      file << mat(i, j) << " ";
    }

    file << std::endl;
  }

  return true;
}


template <typename VectorType, typename VectorIType, typename MatrixType>
void read_vector_from_startfile(const std::string &filename, 
                                size_t num_nodes,
                                const MatrixType &nodes,
                                VectorType &vec1, const char *col_name1,
                                VectorType &vec2, const char *col_name2,
                                VectorType &vec3, const char *col_name3,
                                VectorType &vec4, const char *col_name4,
                                VectorType &vec5, const char *col_name5,
                                VectorType &vec6, const char *col_name6,
                                VectorType &vec7, const char *col_name7,
                                VectorType &vec8, const char *col_name8,
                                VectorType &vec9, const char *col_name9,
                                VectorType &vec10, const char *col_name10,
                                VectorType &vec11, const char *col_name11,
                                VectorIType &vec12, const char *col_name12) {
    std::ifstream file(filename.c_str());
    if (!file) {
        std::cerr << "No start file found!" << std::endl;
    }

    std::string line;
    // Read the header and determine the positions of columns
    std::getline(file, line);
    std::istringstream header(line);
    std::string column;
    std::map<std::string, size_t> column_positions;
    size_t index = 0;
    while (std::getline(header, column, ',')) {
        column_positions[column] = index++;
    }

    // Initialize vectors
    vec1.resize(num_nodes);
    vec2.resize(num_nodes);
    vec3.resize(num_nodes);
    vec4.resize(num_nodes);
    vec5.resize(num_nodes);
    vec6.resize(num_nodes);
    vec7.resize(num_nodes);
    vec8.resize(num_nodes);
    vec9.resize(num_nodes);
    vec10.resize(num_nodes);
    vec11.resize(num_nodes);
    vec12.resize(num_nodes);

      // Read the data
      size_t row = 0;
      while (std::getline(file, line) && row < num_nodes) {
          std::istringstream ss(line);
          std::vector<std::string> data;
          while (std::getline(ss, column, ',')) {
              data.push_back(column);
          }

          // Assign values to vectors
          if (column_positions.find(col_name1) != column_positions.end() && column_positions[col_name1] < data.size()) {
              vec1(row) = std::stod(data[column_positions[col_name1]]);
          }
          if (column_positions.find(col_name1) != column_positions.end() && column_positions[col_name1] < data.size()) {
              vec1(row) = std::stod(data[column_positions[col_name1]]);
          }
          if (column_positions.find(col_name2) != column_positions.end() && column_positions[col_name2] < data.size()) {
              vec2(row) = std::stod(data[column_positions[col_name2]]);
          }
          if (column_positions.find(col_name3) != column_positions.end() && column_positions[col_name3] < data.size()) {
              vec3(row) = std::stod(data[column_positions[col_name3]]);
          }
          if (column_positions.find(col_name4) != column_positions.end() && column_positions[col_name4] < data.size()) {
              vec4(row) = std::stod(data[column_positions[col_name4]]);
          }
          if (column_positions.find(col_name5) != column_positions.end() && column_positions[col_name5] < data.size()) {
              vec5(row) = std::stod(data[column_positions[col_name5]]);
          }
          if (column_positions.find(col_name6) != column_positions.end() && column_positions[col_name6] < data.size()) {
              vec6(row) = std::stod(data[column_positions[col_name6]]);
          }
          if (column_positions.find(col_name7) != column_positions.end() && column_positions[col_name7] < data.size()) {
              vec7(row) = std::stod(data[column_positions[col_name7]]);
          }
          if (column_positions.find(col_name8) != column_positions.end() && column_positions[col_name8] < data.size()) {
              vec8(row) = std::stod(data[column_positions[col_name8]]);
          }
          if (column_positions.find(col_name9) != column_positions.end() && column_positions[col_name9] < data.size()) {
              vec9(row) = std::stod(data[column_positions[col_name9]]);
          }
          if (column_positions.find(col_name10) != column_positions.end() && column_positions[col_name10] < data.size()) {
              vec10(row) = std::stod(data[column_positions[col_name10]]);
          }
          if (column_positions.find(col_name11) != column_positions.end() && column_positions[col_name11] < data.size()) {
              vec11(row) = std::stod(data[column_positions[col_name11]]);
          }
          if (column_positions.find(col_name12) != column_positions.end() && column_positions[col_name12] < data.size()) {
              vec12(row) = std::stoi(data[column_positions[col_name12]]);
          }


          ++row;
      }
  }

bool write_matrix_market_to_file(const std::string &filename, const int n,
                                 const int nnz, const int *row_ptr,
                                 const int *col_ptr, const double *val_ptr) {
  std::ofstream file(filename.c_str());

  if (!file) {
    return false;
  }

  file << "%%MatrixMarket matrix coordinate real general" << std::endl
       << "%Created with utils.h" << std::endl
       << n << " " << n << " " << nnz << std::endl;

  file.precision(16);

  for (int i = 0; i < n; i++) {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
      file << i + 1 << " " << col_ptr[j] + 1 << " " << val_ptr[j] << std::endl;
    }
  }

  return true;
}

template <typename vector_type, typename vector_Itype, typename matrix_type>
bool write_vector_paraview_scalar_to_file(const std::string &filename,
                                          const matrix_type &nodes,
                                          const vector_type &vec1,
                                          const char *col_name1,
                                          const vector_type &vec2,
                                          const char *col_name2,
                                          const vector_type &vec3,
                                          const char *col_name3,
                                          const vector_type &vec4,
                                          const char *col_name4,
                                          const vector_type &vec5,
                                          const char *col_name5,
                                          const vector_type &vec6,
                                          const char *col_name6,
                                          const vector_type &vec7,
                                          const char *col_name7,
                                          const vector_type &vec8,
                                          const char *col_name8,
                                          const vector_type &vec9,
                                          const char *col_name9,
                                          const vector_type &vec10,
                                          const char *col_name10,
                                          const vector_type &vec11,
                                          const char *col_name11,
                                          const vector_Itype &vec12,
                                          const char *col_name12) {
  std::ofstream file(filename.c_str());

  if (!file) {
    return false;
  }

  file.precision(16);

  file << "X,Y,Z," << col_name1 << "," << col_name2 << "," << col_name3 << "," << col_name4 << "," << col_name5 << "," << col_name6
                   << "," << col_name7 << "," << col_name8 << "," << col_name9 << "," << col_name10 << "," << col_name11 << "," << col_name12 
                   << std::endl;

  for (int i = 0; i < vec1.size(); i++) {
    file << nodes(i, 0) << "," << nodes(i, 1) << "," << nodes(i, 2) 
          << "," << vec1[i] << "," << vec2[i] << "," << vec3[i] << "," << vec4[i] << "," << vec5[i] << "," << vec6[i] << "," << vec7[i] << "," << vec8[i]
          << "," << vec9[i] << "," << vec10[i] << "," << vec11[i] << "," << vec12[i]
          << std::endl;
  }

  return true;
}

template <typename vector_type, typename matrix_type>
bool write_vector_paraview_scalar_to_file(const std::string &filename,
                                          const matrix_type &nodes,
                                          const vector_type &vec,
                                          const char *col_name) {
  std::ofstream file(filename.c_str());

  if (!file) {
    return false;
  }

  file.precision(16);

  file << "X,Y,Z,U" << col_name << std::endl;

  for (int i = 0; i < vec.size(); i++) {
    file << nodes(i, 0) << "," << nodes(i, 1) << "," << nodes(i, 2) << "," << vec[i]
         << std::endl;
  }

  return true;
}

template <typename vector_type, typename matrix_type>
bool write_vector_paraview_scalar_to_file(const std::string &filename,
                                          const matrix_type &nodes,
                                          const vector_type &vec,
                                          const char *col_name,
                                          const double flag) {
  std::ofstream file(filename.c_str());

  if (!file) {
    return false;
  }

  file.precision(16);

  file << "X,Y,Z,U" << col_name << std::endl;

  for (int i = 0; i < vec.size(); i++) {
    if (vec[i] <= flag && flag >= 0.0)
      continue;
    else if (vec[i] >= abs(flag) && flag < 0.0)
      continue;
    file << nodes(i, 0) << "," << nodes(i, 1) << "," << nodes(i, 2) << "," << vec[i]
         << std::endl; 
  }

  return true;
}

template <typename vector_type, typename matrix_type>
bool write_vector_paraview_vector_to_file(const std::string &filename,
                                          const matrix_type &nodes,
                                          const vector_type &vec,
                                          const char *col_name_x,
                                          const char *col_name_y,
										  const char *col_name_z) {
  std::ofstream file(filename.c_str());

  if (!file) {
    return false;
  }

  file.precision(16);

  file << "X,Y,Z," << col_name_x << "," << col_name_y << std::endl;

  for (int i = 0; i < vec.size() / 3; i++) {
    file << nodes(i, 0) << "," << nodes(i, 1) << "," << nodes(i, 2);

    for (int j = 0; j < 3; j++) {
      file << "," << vec[3 * i + j];
    }

    file << std::endl;
  }

  return true;
}

//
// Parameters
// A parameter class 
class parameter{
  public:
    int nodesNo;
    int stepsNo;
    int bcNodesNo;
    int bunnyNodesNo;

    double t0;
    double t1;
    double dt;
    double dx;
    double dy;
    double dz;
    double delta;
    double CSolid;
    double CSat;
    double C0Na;
    double C0Ca;
    double C0HCO3;
    double C0HPO4;
    double C0Cl;
    double pKMgOH2;
    double pKMgCO3;
    double pKMgHCO3;
    double pKMgPO4;
    double pKCaCO3;
    double pKCa5OHPO4;
    double pKH20;
    double pKHCO3;
    double pKHPO4;
    double phim;
    double phimse;
    double beta;
    double alphaA;
    double alphaC;
    double k1f;
    double k1b;
    double k2f;
    double k2b;
    double k3f;
    double k3b;
    double k4f;
    double k4b;
    double k5f;
    double k5b;
    double k6f;
    double k6b;
    double k7f;
    double k7b;
    double k8f;
    double k8b;
    double k9f;
    double k9b;
    double JHp;
    double T;
    double R;
    double F;
    double vol_red;

    double z1;
    double D1Solid;
    double D1Liquid;
    double s11;
    double s12;
    double K1prop1;
    double K1prop2;

    double z2;
    double D2Solid;
    double D2Liquid;
    double s21;
    double s22;
    double K2prop1;
    double K2prop2;

    double z3;
    double D3Solid;
    double D3Liquid;
    double s31;
    double s32;
    double K3prop1;
    double K3prop2;

    double z4;
    double D4Solid;
    double D4Liquid;
    double s41;
    double s42;
    double K4prop1;
    double K4prop2;

    double z5;
    double D5Solid;
    double D5Liquid;
    double s51;
    double s52;
    double K5prop1;
    double K5prop2;

    double z6;
    double D6Solid;
    double D6Liquid;
    double s61;
    double s62;
    double K6prop1;
    double K6prop2;

    double z7;
    double D7Solid;
    double D7Liquid;
    double s71;
    double s72;
    double K7prop1;
    double K7prop2;

    double z8;
    double D8Solid;
    double D8Liquid;
    double s81;
    double s82;
    double K8prop1;
    double K8prop2;

    double z9;
    double D9Solid;
    double D9Liquid;
    double s91;
    double s92;
    double K9prop1;
    double K9prop2;

    double z10;
    double D10Solid;
    double D10Liquid;
    double s101;
    double s102;
    double K10prop1;
    double K10prop2;

    double sigma0;
    double sigma1;
    double s111;
    double K11prop1;

    parameter(){
      // Empty
    }

    template <typename vector_Itype>
    void set_int(vector_Itype &controlData1){
      nodesNo = controlData1[0];
      stepsNo = controlData1[1];
      bcNodesNo = controlData1[2];
      bunnyNodesNo = controlData1[3];
    }

    template <typename vector_Dtype>
    void set_double(vector_Dtype &controlData2,  vector_Dtype &controlDataC1, vector_Dtype &controlDataC2, 
                    vector_Dtype &controlDataC3, vector_Dtype &controlDataC4, vector_Dtype &controlDataC5, 
                    vector_Dtype &controlDataC6, vector_Dtype &controlDataC7, vector_Dtype &controlDataC8, 
                    vector_Dtype &controlDataC9, vector_Dtype &controlDataC10, vector_Dtype &controlDataC11)
                    {
      t0 = controlData2[0];
      t1 = controlData2[1];
      dt = controlData2[2];
      dx = controlData2[3]; 
      dy = controlData2[4];
      dz = controlData2[5];
      delta = controlData2[6];
      CSolid = controlData2[7];
      CSat = controlData2[8];
      C0Na = controlData2[9];
      C0Ca = controlData2[10];
      C0HCO3 = controlData2[11];
      C0HPO4 = controlData2[12];
      C0Cl = controlData2[13];
      pKMgOH2 = controlData2[14]; 
      pKMgCO3 = controlData2[15]; 
      pKMgHCO3 = controlData2[16]; 
      pKMgPO4 = controlData2[17]; 
      pKCaCO3 = controlData2[18]; 
      pKCa5OHPO4 = controlData2[19];
      pKH20 = controlData2[20]; 
      pKHCO3 = controlData2[21]; 
      pKHPO4 = controlData2[22]; 
      phim = controlData2[23]; 
      phimse = controlData2[24];
      beta = controlData2[25];
      alphaA = controlData2[26];
      alphaC = controlData2[27];
      k1f = controlData2[28];
      k1b = controlData2[29]; 
      k2f = controlData2[30]; 
      k2b = controlData2[31]; 
      k3f = controlData2[32]; 
      k3b = controlData2[33]; 
      k4f = controlData2[34]; 
      k4b = controlData2[35]; 
      k5f = controlData2[36]; 
      k5b = controlData2[37]; 
      k6f = controlData2[38];
      k6b = controlData2[39];
      k7f = controlData2[40]; 
      k7b = controlData2[41]; 
      k8f = controlData2[42]; 
      k8b = controlData2[43]; 
      k9f = controlData2[44]; 
      k9b = controlData2[45]; 
      JHp = controlData2[46]; 
      T = controlData2[47]; 
      R = controlData2[48]; 
      F = controlData2[49];
      vol_red = controlData2[50];

      z1 = controlDataC1[0];
      D1Solid = controlDataC1[1];
      D1Liquid = controlDataC1[2];
      s11 = controlDataC1[3];
      s12 = controlDataC1[4];
      K1prop1 = controlDataC1[5];
      K1prop2 = controlDataC1[6];

      z2 = controlDataC2[0];
      D2Solid = controlDataC2[1];
      D2Liquid = controlDataC2[2];
      s21 = controlDataC2[3];
      s22 = controlDataC2[4];
      K2prop1 = controlDataC2[5];
      K2prop2 = controlDataC2[6];

      z3 = controlDataC3[0];
      D3Solid = controlDataC3[1];
      D3Liquid = controlDataC3[2];
      s31 = controlDataC3[3];
      s32 = controlDataC3[4];
      K3prop1 = controlDataC3[5];
      K3prop2 = controlDataC3[6];

      z4 = controlDataC4[0];
      D4Solid = controlDataC4[1];
      D4Liquid = controlDataC4[2];
      s41 = controlDataC4[3];
      s42 = controlDataC4[4];
      K4prop1 = controlDataC4[5];
      K4prop2 = controlDataC4[6];

      z5 = controlDataC5[0];
      D5Solid = controlDataC5[1];
      D5Liquid = controlDataC5[2];
      s51 = controlDataC5[3];
      s52 = controlDataC5[4];
      K5prop1 = controlDataC5[5];
      K5prop2 = controlDataC5[6];

      z6 = controlDataC6[0];
      D6Solid = controlDataC6[1];
      D6Liquid = controlDataC6[2];
      s61 = controlDataC6[3];
      s62 = controlDataC6[4];
      K6prop1 = controlDataC6[5];
      K6prop2 = controlDataC6[6];

      z7 = controlDataC7[0];
      D7Solid = controlDataC7[1];
      D7Liquid = controlDataC7[2];
      s71 = controlDataC7[3];
      s72 = controlDataC7[4];
      K7prop1 = controlDataC7[5];
      K7prop2 = controlDataC7[6];

      z8 = controlDataC8[0];
      D8Solid = controlDataC8[1];
      D8Liquid = controlDataC8[2];
      s81 = controlDataC8[3];
      s82 = controlDataC8[4];
      K8prop1 = controlDataC8[5];
      K8prop2 = controlDataC8[6];

      z9 = controlDataC9[0];
      D9Solid = controlDataC9[1];
      D9Liquid = controlDataC9[2];
      s91 = controlDataC9[3];
      s92 = controlDataC9[4];
      K9prop1 = controlDataC9[5];
      K9prop2 = controlDataC9[6];

      z10 = controlDataC10[0];
      D10Solid = controlDataC10[1];
      D10Liquid = controlDataC10[2];
      s101 = controlDataC10[3];
      s102 = controlDataC10[4];
      K10prop1 = controlDataC10[5];
      K10prop2 = controlDataC10[6];

      sigma0 = controlDataC11[0];
      sigma1 = controlDataC11[1];
      s111 = controlDataC11[2];
      K11prop1 = controlDataC11[3];
    }
};

//
// timer
// A class for accurate timing

class timer {
 public:
  timer() {
    // Nothing to do!
  }

  void tic() {
#ifdef _WIN32
    QueryPerformanceCounter(&start);
#else
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
  }

  double toc() {
#ifdef _WIN32
    LARGE_INTEGER stop;
    LARGE_INTEGER frequency;

    QueryPerformanceCounter(&stop);
    QueryPerformanceFrequency(&frequency);

    return (stop.QuadPart - start.QuadPart) /
           static_cast<double>(frequency.QuadPart);
#else
    struct timespec stop;

    clock_gettime(CLOCK_MONOTONIC, &stop);

    return static_cast<double>(((unsigned long long)stop.tv_sec *
                                    (1000ULL * 1000ULL * 1000ULL) +
                                (unsigned long long)stop.tv_nsec) -
                               ((unsigned long long)start.tv_sec *
                                    (1000ULL * 1000ULL * 1000ULL) +
                                (unsigned long long)start.tv_nsec)) /
           1.00e09;
#endif
  }

 private:
#ifdef _WIN32
  LARGE_INTEGER start;
#else
  struct timespec start;
#endif
};

}  // namespace utils
