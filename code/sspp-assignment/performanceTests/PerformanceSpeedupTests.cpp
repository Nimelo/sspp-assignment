#include "PerformanceSpeedupTest.h"
#include <gtest/gtest.h>
#include "TestMacros.h"
#include "TestingUtilities.h"
#include "CRSSolver.h"
#include "CRSOpenMPSolver.h"
#include "CRSCudaSolver.h"
#include "EllpackSolver.h"
#include "ELLPACKOpenMPSolver.h"
#include "ELLPACKCudaSolver.h"

#define ITERATIONS 100

#define PRINT_DOUBLE_FLOAT_COMPARISON(RESULT)                                                 \
TEST_COUT << "Float: " << RESULT.GetFloatTime() << "s\n";                                     \
TEST_COUT << "Double: " << RESULT.GetDoubleTime() << "s\n";                                   \
TEST_COUT << "FLOPS: " << RESULT.GetFlops() << "\n";                                          \
TEST_COUT << "DP FLOPS: " << RESULT.GetDPFlops() << "\n";                                     \
TEST_COUT << "Float/double ratio: " << RESULT.GetFloatTime() / RESULT.GetDoubleTime() << "\n";\
TEST_COUT << "Solution norm: " << RESULT.GetResultNorm() << std::endl                         \

TEST_F(PerformanceSpeedupTest, ALWAYS_TRUE) {
  TEST_COUT << "Hello world!\n";
}

TEST_F(ABCDE, SERIAL_CRS_FLOAT_VS_DOUBLE) {
  sspp::common::CRSSolver<float> solver_float;
  sspp::common::CRSSolver<double> solver_double;
  
  DoubleFloatComparison comparison = this->FloatDoubleCRSComparison(solver_float, solver_double, ITERATIONS);
  PRINT_DOUBLE_FLOAT_COMPARISON(comparison);
}

TEST_F(ABCDE, OPENMP_CRS_FLOAT_VS_DOUBLE) {
  omp_set_num_threads(2);
  sspp::openmp::CRSOpenMPSolver<float> solver_float;
  sspp::openmp::CRSOpenMPSolver<double> solver_double;

  DoubleFloatComparison comparison = this->FloatDoubleCRSComparison(solver_float, solver_double, ITERATIONS);
  PRINT_DOUBLE_FLOAT_COMPARISON(comparison);
}

TEST_F(ABCDE, CUDA_CRS_FLOAT_VS_DOUBLE) {
  sspp::cuda::CRSCudaSolver<float> solver_float;
  sspp::cuda::CRSCudaSolver<double> solver_double;

  DoubleFloatComparison comparison = this->FloatDoubleCRSComparison(solver_float, solver_double, ITERATIONS);
  PRINT_DOUBLE_FLOAT_COMPARISON(comparison);
}

TEST_F(ABCDE, SERIAL_ELLPACK_FLOAT_VS_DOUBLE) {
  sspp::common::ELLPACKSolver<float> solver_float;
  sspp::common::ELLPACKSolver<double> solver_double;

  DoubleFloatComparison comparison = this->FloatDoubleELLPACKComparison(solver_float, solver_double, ITERATIONS);
  PRINT_DOUBLE_FLOAT_COMPARISON(comparison);
}

TEST_F(ABCDE, OPENMP_ELLPACK_FLOAT_VS_DOUBLE) {
  omp_set_num_threads(2);
  sspp::openmp::ELLPACKOpenMPSolver<float> solver_float;
  sspp::openmp::ELLPACKOpenMPSolver<double> solver_double;

  DoubleFloatComparison comparison = this->FloatDoubleELLPACKComparison(solver_float, solver_double, ITERATIONS);
  PRINT_DOUBLE_FLOAT_COMPARISON(comparison);
}

TEST_F(ABCDE, CUDA_ELLPACK_FLOAT_VS_DOUBLE) {
  sspp::cuda::ELLPACKCudaSolver<float> solver_float;
  sspp::cuda::ELLPACKCudaSolver<double> solver_double;

  DoubleFloatComparison comparison = this->FloatDoubleELLPACKComparison(solver_float, solver_double, ITERATIONS);
  PRINT_DOUBLE_FLOAT_COMPARISON(comparison);
}