#pragma once
#include <gtest/gtest.h>
#include "TestStream.h"
#include "CRSSolver.h"
#include "CRSOpenMPSolver.h"
#include "CRSCudaSolver.h"
#include "EllpackSolver.h"
#include "ELLPACKOpenMPSolver.h"
#include "ELLPACKCudaSolver.h"
#include "MatrixTestInterface.h"

#define ITERATIONS 100

#define CREATE_FIXTURE_AND_TESTS(FIXTURE_NAME, KEY, PATH) \
        CREATE_FIXTURE(FIXTURE_NAME, KEY, PATH)           \
        CREATE_PERFORMANCE_CUMULATIVE_F(FIXTURE_NAME)     \
        CREATE_FLOAT_DOUBLE_TEST_F(FIXTURE_NAME)          \
        CREATE_SPEEDUP_TEST_F(FIXTURE_NAME)               \

#pragma warning (push)
#pragma warning( disable : 4356)
#define CREATE_FIXTURE(FIXTURE_NAME, KEY, PATH)   \
class FIXTURE_NAME : public MatrixTestInterface { \
protected:                                        \
   std::string GetPath() {                        \
    return PATH;                                  \
  };                                              \
  std::string GetKey() {                          \
    return KEY;                                   \
  };                                              \
};                                                \

#pragma warning(pop)

#define CREATE_PERFORMANCE_CUMULATIVE_F(FIXTURE_NAME) \
        CREATE_CUDA_CUMULATIVE_SPEEDUP(FIXTURE_NAME)  \
        CREATE_OPENMP_CUMULATIVE_SPEEDUP(FIXTURE_NAME)\

#define CREATE_PERFORMANCE_TEST_F(FIXTURE_NAME)  \
        CREATE_FLOAT_DOUBLE_TEST_F(FIXTURE_NAME) \
        CREATE_SPEEDUP_TEST_F(FIXTURE_NAME)      \

#define CREATE_FLOAT_DOUBLE_TEST_F(FIXTURE_NAME)         \
        CREATE_FLOAT_DOUBLE_SERIAL_TEST_F(FIXTURE_NAME)  \
        CREATE_FLOAT_DOUBLE_CUDA_TEST_F(FIXTURE_NAME)    \
        CREATE_FLOAT_DOUBLE_OPENMP_TEST_F(FIXTURE_NAME)  \

#define CREATE_SPEEDUP_TEST_F(FIXTURE_NAME)        \
        CREATE_SPEEDUP_CUDA_TEST_F(FIXTURE_NAME)   \
        CREATE_SPEEDUP_OPENMP_TEST_F(FIXTURE_NAME) \

#define CREATE_FLOAT_DOUBLE_SERIAL_TEST_F(FIXTURE_NAME)                                                          \
TEST_F(FIXTURE_NAME, SERIAL_CRS_FLOAT_VS_DOUBLE) {                                                               \
  sspp::common::CRSSolver<float> solver_float;                                                                   \
  sspp::common::CRSSolver<double> solver_double;                                                                 \
                                                                                                                 \
  DoubleFloatComparison comparison = this->FloatDoubleCRSComparison(solver_float, solver_double, ITERATIONS);    \
  PRINT_DOUBLE_FLOAT_COMPARISON(comparison);                                                                     \
}                                                                                                                \
                                                                                                                 \
TEST_F(FIXTURE_NAME, SERIAL_ELLPACK_FLOAT_VS_DOUBLE) {                                                           \
  sspp::common::ELLPACKSolver<float> solver_float;                                                               \
  sspp::common::ELLPACKSolver<double> solver_double;                                                             \
                                                                                                                 \
  DoubleFloatComparison comparison = this->FloatDoubleELLPACKComparison(solver_float, solver_double, ITERATIONS);\
  PRINT_DOUBLE_FLOAT_COMPARISON(comparison);                                                                     \
}                                                                                                                \

#define CREATE_FLOAT_DOUBLE_CUDA_TEST_F(FIXTURE_NAME)                                                              \
TEST_F(FIXTURE_NAME, CUDA_CRS_FLOAT_VS_DOUBLE) {                                                                   \
  sspp::cuda::CRSCudaSolver<float> solver_float;                                                                   \
  sspp::cuda::CRSCudaSolver<double> solver_double;                                                                 \
                                                                                                                   \
  DoubleFloatComparison comparison = this->FloatDoubleCRSComparison(solver_float, solver_double, ITERATIONS);      \
  PRINT_DOUBLE_FLOAT_COMPARISON(comparison);                                                                       \
}                                                                                                                  \
                                                                                                                   \
TEST_F(FIXTURE_NAME, CUDA_ELLPACK_FLOAT_VS_DOUBLE) {                                                               \
    sspp::cuda::ELLPACKCudaSolver<float> solver_float;                                                             \
    sspp::cuda::ELLPACKCudaSolver<double> solver_double;                                                           \
                                                                                                                   \
    DoubleFloatComparison comparison = this->FloatDoubleELLPACKComparison(solver_float, solver_double, ITERATIONS);\
    PRINT_DOUBLE_FLOAT_COMPARISON(comparison);                                                                     \
}                                                                                                                  \

#define CREATE_FLOAT_DOUBLE_OPENMP_TEST_F(FIXTURE_NAME)                                                          \
TEST_F(FIXTURE_NAME, OPENMP_ELLPACK_FLOAT_VS_DOUBLE) {                                                           \
  sspp::openmp::ELLPACKOpenMPSolver<float> solver_float;                                                         \
  sspp::openmp::ELLPACKOpenMPSolver<double> solver_double;                                                       \
  solver_float.SetThreads(2);                                                                                    \
  solver_double.SetThreads(2);                                                                                   \
                                                                                                                 \
  DoubleFloatComparison comparison = this->FloatDoubleELLPACKComparison(solver_float, solver_double, ITERATIONS);\
  PRINT_DOUBLE_FLOAT_COMPARISON(comparison);                                                                     \
}                                                                                                                \
                                                                                                                 \
TEST_F(FIXTURE_NAME, OPENMP_CRS_FLOAT_VS_DOUBLE) {                                                               \
  sspp::openmp::CRSOpenMPSolver<float> solver_float;                                                             \
  sspp::openmp::CRSOpenMPSolver<double> solver_double;                                                           \
  solver_float.SetThreads(2);                                                                                    \
  solver_double.SetThreads(2);                                                                                   \
                                                                                                                 \
  DoubleFloatComparison comparison = this->FloatDoubleCRSComparison(solver_float, solver_double, ITERATIONS);    \
  PRINT_DOUBLE_FLOAT_COMPARISON(comparison);                                                                     \
}                                                                                                                \

#define CREATE_SPEEDUP_CUDA_TEST_F(FIXTURE_NAME)                                         \
TEST_F(FIXTURE_NAME, CUDA_CRS_FLOAT_SPEEDUP) {                                           \
  sspp::common::CRSSolver<float> serial_solver;                                          \
  sspp::cuda::CRSCudaSolver<float> cuda_solver;                                          \
  auto comparison = this->SpeedupCRS<float>(serial_solver, cuda_solver, ITERATIONS);     \
                                                                                         \
  PRINT_SERIAL_PARALLEL_COMPARISON(comparison);                                          \
}                                                                                        \
                                                                                         \
TEST_F(FIXTURE_NAME, CUDA_CRS_DOUBLE_SPEEDUP) {                                          \
  sspp::common::CRSSolver<double> serial_solver;                                         \
  sspp::cuda::CRSCudaSolver<double> cuda_solver;                                         \
  auto comparison = this->SpeedupCRS<double>(serial_solver, cuda_solver, ITERATIONS);    \
                                                                                         \
  PRINT_SERIAL_PARALLEL_COMPARISON(comparison);                                          \
}                                                                                        \
                                                                                         \
                                                                                         \
TEST_F(FIXTURE_NAME, CUDA_ELLPACK_FLOAT_SPEEDUP) {                                       \
  sspp::common::ELLPACKSolver<float> serial_solver;                                      \
  sspp::cuda::ELLPACKCudaSolver<float> cuda_solver;                                      \
  auto comparison = this->SpeedupEllpack<float>(serial_solver, cuda_solver, ITERATIONS); \
                                                                                         \
  PRINT_SERIAL_PARALLEL_COMPARISON(comparison);                                          \
}                                                                                        \
                                                                                         \
TEST_F(FIXTURE_NAME, CUDA_ELLPACK_DOUBLE_SPEEDUP) {                                      \
  sspp::common::ELLPACKSolver<double> serial_solver;                                     \
  sspp::cuda::ELLPACKCudaSolver<double> cuda_solver;                                     \
  auto comparison = this->SpeedupEllpack<double>(serial_solver, cuda_solver, ITERATIONS);\
                                                                                         \
  PRINT_SERIAL_PARALLEL_COMPARISON(comparison);                                          \
}                                                                                        \

#define CREATE_SPEEDUP_OPENMP_TEST_F(FIXTURE_NAME)                                                         \
TEST_F(FIXTURE_NAME, OPENMP_CRS_FLOAT_SPEEDUP) {                                                           \
  sspp::common::CRSSolver<float> serial_solver;                                                            \
  sspp::openmp::CRSOpenMPSolver<float> parallel_solver;                                                    \
                                                                                                           \
  auto best = this->CheckPerformance([&serial_solver, &parallel_solver, this]()->SerialParallelComparison {\
    return this->SpeedupCRS(serial_solver, parallel_solver, ITERATIONS);                                   \
  });                                                                                                      \
                                                                                                           \
  PRINT_OPENMP_THREADS(best);                                                                              \
}                                                                                                          \
                                                                                                           \
TEST_F(FIXTURE_NAME, OPENMP_CRS_DOUBLE_SPEEDUP) {                                                          \
  sspp::common::CRSSolver<double> serial_solver;                                                           \
  sspp::openmp::CRSOpenMPSolver<double> parallel_solver;                                                   \
                                                                                                           \
  auto best = this->CheckPerformance([&serial_solver, &parallel_solver, this]()->SerialParallelComparison {\
    return this->SpeedupCRS(serial_solver, parallel_solver, ITERATIONS);                                   \
  });                                                                                                      \
                                                                                                           \
  PRINT_OPENMP_THREADS(best);                                                                              \
}                                                                                                          \
                                                                                                           \
TEST_F(FIXTURE_NAME, OPENMP_ELLPACK_FLOAT_SPEEDUP) {                                                       \
  sspp::common::ELLPACKSolver<float> serial_solver;                                                        \
  sspp::openmp::ELLPACKOpenMPSolver<float> parallel_solver;                                                \
                                                                                                           \
  auto best = this->CheckPerformance([&serial_solver, &parallel_solver, this]()->SerialParallelComparison {\
    return this->SpeedupEllpack(serial_solver, parallel_solver, ITERATIONS);                               \
  });                                                                                                      \
                                                                                                           \
  PRINT_OPENMP_THREADS(best);                                                                              \
}                                                                                                          \
                                                                                                           \
TEST_F(FIXTURE_NAME, OPENMP_ELLPACK_DOUBLE_SPEEDUP) {                                                      \
  sspp::common::ELLPACKSolver<double> serial_solver;                                                       \
  sspp::openmp::ELLPACKOpenMPSolver<double> parallel_solver;                                               \
                                                                                                           \
  auto best = this->CheckPerformance([&serial_solver, &parallel_solver, this]()->SerialParallelComparison {\
    return this->SpeedupEllpack(serial_solver, parallel_solver, ITERATIONS);                               \
  });                                                                                                      \
                                                                                                           \
  PRINT_OPENMP_THREADS(best);                                                                              \
}                                                                                                          \

#define CREATE_OPENMP_CUMULATIVE_SPEEDUP(FIXTURE_NAME)                           \
TEST_F(FIXTURE_NAME, OPENMP_CUMULATIVE_SPEEDUP) {                                \
  sspp::common::ELLPACKSolver<float> serial_f_ellpack;                           \
  sspp::common::ELLPACKSolver<double> serial_d_ellpack;                          \
  sspp::openmp::ELLPACKOpenMPSolver<float> openmp_f_ellpack;                     \
  sspp::openmp::ELLPACKOpenMPSolver<double> openmp_d_ellpack;                    \
  sspp::common::CRSSolver<float> serial_f_crs;                                   \
  sspp::common::CRSSolver<double> serial_d_crs;                                  \
  sspp::openmp::CRSOpenMPSolver<float> openmp_f_crs;                             \
  sspp::openmp::CRSOpenMPSolver<double> openmp_d_crs;                            \
                                                                                 \
  TEST_COUT << CumlativeOpenMPPerformance(                                       \
    [&serial_f_crs, &openmp_f_crs, this]() -> SerialParallelComparison {         \
    return SpeedupCRS(serial_f_crs, openmp_f_crs, ITERATIONS);                   \
  }, [&serial_d_crs, &openmp_d_crs, this]() -> SerialParallelComparison {        \
    return SpeedupCRS(serial_d_crs, openmp_d_crs, ITERATIONS);                   \
  }, [&serial_f_ellpack, &openmp_f_ellpack, this]() -> SerialParallelComparison {\
    return SpeedupEllpack(serial_f_ellpack, openmp_f_ellpack, ITERATIONS);       \
  }, [&serial_d_ellpack, &openmp_d_ellpack, this]() -> SerialParallelComparison {\
    return SpeedupEllpack(serial_d_ellpack, openmp_d_ellpack, ITERATIONS);       \
  }                                                                              \
  ) << std::endl;                                                                \
}                                                                                \

#define CREATE_CUDA_CUMULATIVE_SPEEDUP(FIXTURE_NAME)                             \
TEST_F(FIXTURE_NAME, CUDA_CUMULATIVE_SPEEDUP) {                                  \
  sspp::common::ELLPACKSolver<float> serial_f_ellpack;                           \
  sspp::common::ELLPACKSolver<double> serial_d_ellpack;                          \
  sspp::cuda::ELLPACKCudaSolver<float> openmp_f_ellpack;                         \
  sspp::cuda::ELLPACKCudaSolver<double> openmp_d_ellpack;                        \
  sspp::common::CRSSolver<float> serial_f_crs;                                   \
  sspp::common::CRSSolver<double> serial_d_crs;                                  \
  sspp::cuda::CRSCudaSolver<float> openmp_f_crs;                                 \
  sspp::cuda::CRSCudaSolver<double> openmp_d_crs;                                \
                                                                                 \
  TEST_COUT << CumulativeCudaPerformance(                                        \
    [&serial_f_crs, &openmp_f_crs, this]() -> SerialParallelComparison {         \
    return SpeedupCRS(serial_f_crs, openmp_f_crs, ITERATIONS);                   \
  }, [&serial_d_crs, &openmp_d_crs, this]() -> SerialParallelComparison {        \
    return SpeedupCRS(serial_d_crs, openmp_d_crs, ITERATIONS);                   \
  }, [&serial_f_ellpack, &openmp_f_ellpack, this]() -> SerialParallelComparison {\
    return SpeedupEllpack(serial_f_ellpack, openmp_f_ellpack, ITERATIONS);       \
  }, [&serial_d_ellpack, &openmp_d_ellpack, this]() -> SerialParallelComparison {\
    return SpeedupEllpack(serial_d_ellpack, openmp_d_ellpack, ITERATIONS);       \
  }                                                                              \
  ) << std::endl;                                                                \
}                                                                                \

#define CREATE_FAKE_TEST_CASE(FIXTURE_NAME) \
TEST_F(FIXTURE_NAME, EMPTY_CASE) { }        \
;
;