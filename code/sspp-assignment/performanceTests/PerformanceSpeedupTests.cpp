#include "PerformanceSpeedupTest.h"
#include <gtest/gtest.h>
#include "TestMacros.h"
#include "TestStream.h"
#include "CRSSolver.h"
#include "CRSOpenMPSolver.h"
#include "CRSCudaSolver.h"
#include "EllpackSolver.h"
#include "ELLPACKOpenMPSolver.h"
#include "ELLPACKCudaSolver.h"

#define ITERATIONS 100

TEST_F(FIXTURE_NAME, SERIAL_CRS_FLOAT_VS_DOUBLE) {
  sspp::common::CRSSolver<float> solver_float;
  sspp::common::CRSSolver<double> solver_double;

  DoubleFloatComparison comparison = this->FloatDoubleCRSComparison(solver_float, solver_double, ITERATIONS);
  PRINT_DOUBLE_FLOAT_COMPARISON(comparison);
}

TEST_F(FIXTURE_NAME, OPENMP_CRS_FLOAT_VS_DOUBLE) {
  omp_set_num_threads(2);
  sspp::openmp::CRSOpenMPSolver<float> solver_float;
  sspp::openmp::CRSOpenMPSolver<double> solver_double;

  DoubleFloatComparison comparison = this->FloatDoubleCRSComparison(solver_float, solver_double, ITERATIONS);
  PRINT_DOUBLE_FLOAT_COMPARISON(comparison);
}

TEST_F(FIXTURE_NAME, CUDA_CRS_FLOAT_VS_DOUBLE) {
  sspp::cuda::CRSCudaSolver<float> solver_float;
  sspp::cuda::CRSCudaSolver<double> solver_double;

  DoubleFloatComparison comparison = this->FloatDoubleCRSComparison(solver_float, solver_double, ITERATIONS);
  PRINT_DOUBLE_FLOAT_COMPARISON(comparison);
}

TEST_F(FIXTURE_NAME, SERIAL_ELLPACK_FLOAT_VS_DOUBLE) {
  sspp::common::ELLPACKSolver<float> solver_float;
  sspp::common::ELLPACKSolver<double> solver_double;

  DoubleFloatComparison comparison = this->FloatDoubleELLPACKComparison(solver_float, solver_double, ITERATIONS);
  PRINT_DOUBLE_FLOAT_COMPARISON(comparison);
}

TEST_F(FIXTURE_NAME, OPENMP_ELLPACK_FLOAT_VS_DOUBLE) {
  omp_set_num_threads(2);
  sspp::openmp::ELLPACKOpenMPSolver<float> solver_float;
  sspp::openmp::ELLPACKOpenMPSolver<double> solver_double;

  DoubleFloatComparison comparison = this->FloatDoubleELLPACKComparison(solver_float, solver_double, ITERATIONS);
  PRINT_DOUBLE_FLOAT_COMPARISON(comparison);
}

TEST_F(FIXTURE_NAME, CUDA_ELLPACK_FLOAT_VS_DOUBLE) {
  sspp::cuda::ELLPACKCudaSolver<float> solver_float;
  sspp::cuda::ELLPACKCudaSolver<double> solver_double;

  DoubleFloatComparison comparison = this->FloatDoubleELLPACKComparison(solver_float, solver_double, ITERATIONS);
  PRINT_DOUBLE_FLOAT_COMPARISON(comparison);
}

TEST_F(FIXTURE_NAME, CUDA_CRS_FLOAT_SPEEDUP) {
  sspp::common::CRSSolver<float> serial_solver;
  sspp::cuda::CRSCudaSolver<float> cuda_solver;
  auto comparison = this->SpeedupCRS<float>(serial_solver, cuda_solver, ITERATIONS);

  PRINT_SERIAL_PARALLEL_COMPARISON(comparison);
}

TEST_F(FIXTURE_NAME, CUDA_CRS_DOUBLE_SPEEDUP) {
  sspp::common::CRSSolver<double> serial_solver;
  sspp::cuda::CRSCudaSolver<double> cuda_solver;
  auto comparison = this->SpeedupCRS<double>(serial_solver, cuda_solver, ITERATIONS);

  PRINT_SERIAL_PARALLEL_COMPARISON(comparison);
}

TEST_F(FIXTURE_NAME, CUDA_ELLPACK_FLOAT_SPEEDUP) {
  sspp::common::ELLPACKSolver<float> serial_solver;
  sspp::cuda::ELLPACKCudaSolver<float> cuda_solver;
  auto comparison = this->SpeedupEllpack<float>(serial_solver, cuda_solver, ITERATIONS);

  PRINT_SERIAL_PARALLEL_COMPARISON(comparison);
}

TEST_F(FIXTURE_NAME, CUDA_ELLPACK_DOUBLE_SPEEDUP) {
  sspp::common::ELLPACKSolver<double> serial_solver;
  sspp::cuda::ELLPACKCudaSolver<double> cuda_solver;
  auto comparison = this->SpeedupEllpack<double>(serial_solver, cuda_solver, ITERATIONS);

  PRINT_SERIAL_PARALLEL_COMPARISON(comparison);
}

TEST_F(FIXTURE_NAME, OPENMP_CRS_FLOAT_SPEEDUP) {
  sspp::common::CRSSolver<float> serial_solver;
  sspp::openmp::CRSOpenMPSolver<float> parallel_solver;

  auto best = this->CheckPerformance([&serial_solver, &parallel_solver, this]()->SerialParallelComparison {
    return this->SpeedupCRS(serial_solver, parallel_solver, ITERATIONS);
  });

  PRINT_OPENMP_THREADS(best);
}

TEST_F(FIXTURE_NAME, OPENMP_CRS_DOUBLE_SPEEDUP) {
  sspp::common::CRSSolver<double> serial_solver;
  sspp::openmp::CRSOpenMPSolver<double> parallel_solver;

  auto best = this->CheckPerformance([&serial_solver, &parallel_solver, this]()->SerialParallelComparison {
    return this->SpeedupCRS(serial_solver, parallel_solver, ITERATIONS);
  });

  PRINT_OPENMP_THREADS(best);
}

TEST_F(FIXTURE_NAME, OPENMP_ELLPACK_FLOAT_SPEEDUP) {
  sspp::common::ELLPACKSolver<float> serial_solver;
  sspp::openmp::ELLPACKOpenMPSolver<float> parallel_solver;

  auto best = this->CheckPerformance([&serial_solver, &parallel_solver, this]()->SerialParallelComparison {
    return this->SpeedupEllpack(serial_solver, parallel_solver, ITERATIONS);
  });

  PRINT_OPENMP_THREADS(best);
}

TEST_F(FIXTURE_NAME, OPENMP_ELLPACK_DOUBLE_SPEEDUP) {
  sspp::common::ELLPACKSolver<double> serial_solver;
  sspp::openmp::ELLPACKOpenMPSolver<double> parallel_solver;

  auto best = this->CheckPerformance([&serial_solver, &parallel_solver, this]()->SerialParallelComparison {
    return this->SpeedupEllpack(serial_solver, parallel_solver, ITERATIONS);
  });

  PRINT_OPENMP_THREADS(best);
}