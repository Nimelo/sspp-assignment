#pragma once

#include <gtest/gtest.h>
#include <functional>
#include <vector>
#include <cmath>
#include "ELLPACK.h"
#include "CRS.h"
#include "ChronoStopwatch.h"
#include "TestStream.h"
#include "MatrixContainer.h"
#include "DoubleFloatComparisonResult.h"
#include "AbstractCRSSolver.h"
#include "AbstractELLPACKSolver.h"
#include "SerialParallelComparison.h"
#include "OpenMPPerformance.h"
#include <omp.h>
#include "OpenMPCumulativeResult.h"
#include "CudaCumulativeResult.h"

#define PRINT_DOUBLE_FLOAT_COMPARISON(RESULT)                                                 \
TEST_COUT << "Float: " << RESULT.GetFloatTime() << "s\n";                                     \
TEST_COUT << "Double: " << RESULT.GetDoubleTime() << "s\n";                                   \
TEST_COUT << "FLOPS: " << RESULT.GetFlops() << "\n";                                          \
TEST_COUT << "DP FLOPS: " << RESULT.GetDPFlops() << "\n";                                     \
TEST_COUT << "Float/double ratio: " << RESULT.GetFloatTime() / RESULT.GetDoubleTime() << "\n";\
TEST_COUT << "Solution norm: " << RESULT.GetResultNorm() << std::endl                         \

#define PRINT_SERIAL_PARALLEL_COMPARISON(comparison)                   \
TEST_COUT << "Serial: " << comparison.GetSerialTime() << '\n';         \
TEST_COUT << "Parallel: " << comparison.GetParallelTime() << '\n';     \
TEST_COUT << std::string(typeid(comparison.GetSerialOps()).name())     \
+ " FLOPS Serial: " << comparison.GetSerialOps() << '\n';              \
TEST_COUT << std::string(typeid(comparison.GetSerialOps()).name())     \
+  " FLOPS Parallel: " << comparison.GetParallelOps() << '\n';         \
TEST_COUT << "Speedup: " << comparison.GetSpeedup() << '\n';           \
TEST_COUT << "Solution norm: " << comparison.GetSolutionNorm() << '\n' \

#define PRINT_OPENMP_THREADS(result)                          \
TEST_COUT << '\n';                                            \
TEST_COUT << "Threads: " << result.GetThreadsNumber() << '\n';\
TEST_COUT << "Speedup: " << result.GetSpeedup() << '\n';      \

class MatrixTestInterface : public ::testing::Test {
public:
  void SetUp() {
    using namespace sspp::common;
    static_key_ = GetKey();
    if(MatrixContainer::GetInstance().Exist(GetKey())) {
      TEST_COUT << "Matrix with key: " << GetKey() << " already loaded." << std::endl;
    } else {
      TEST_COUT << "Reading matrix-market from: " << GetPath() << std::endl;
      stopwatch_.Start();
      MatrixContainer::GetInstance().LoadOnce(GetPath(), GetKey());
      stopwatch_.Stop();
      TEST_COUT << "Finished in: " << stopwatch_.GetElapsedSeconds() << "s" << std::endl;
    }
  }

  static void TearDownTestCase() {
    TEST_COUT << "Deleting matrix with key: " << static_key_ << std::endl;
    MatrixContainer::GetInstance().Delete(static_key_);
  }
protected:
  OpenMPCumulativeResult CumlativeOpenMPPerformance(std::function<SerialParallelComparison()> float_crs,
                                                    std::function<SerialParallelComparison()> double_crs,
                                                    std::function<SerialParallelComparison()> float_ellpack,
                                                    std::function<SerialParallelComparison()> double_ellpack) {
    OpenMPCumulativeResult result;
    unsigned long long max_avalaible_threads = omp_get_max_threads();
    TEST_COUT << "Max avalaible threads: " << max_avalaible_threads << '\n';
    for(unsigned long long i = 2; i <= max_avalaible_threads; i++) {
      TEST_COUT << "Iteration for: " << i << " threads." << '\n';
      omp_set_num_threads(i);
      result.InsertResult(i, float_crs(), double_crs(), float_ellpack(), double_ellpack());
    }
    std::ofstream os(GetKey() + "openmp.performance");
    os << result;
    os.close();
    return result;
  }

  CudaCumulativeResult CumulativeCudaPerformance(std::function<SerialParallelComparison()> float_crs,
                                                 std::function<SerialParallelComparison()> double_crs,
                                                 std::function<SerialParallelComparison()> float_ellpack,
                                                 std::function<SerialParallelComparison()> double_ellpack) {
    TEST_COUT << "FLOAT CRS" << std::endl;
    auto float_crs_result = float_crs();
    TEST_COUT << "DOUBLE CRS" << std::endl;
    auto double_crs_result = double_crs();
    TEST_COUT << "FLOAT ELLPACK" << std::endl;
    auto float_ellpack_result = float_ellpack();
    TEST_COUT << "DOUBLE ELLPACK" << std::endl;
    auto double_ellpack_result = double_ellpack();
    auto result = CudaCumulativeResult(float_crs_result, double_crs_result, float_ellpack_result, double_ellpack_result);

    std::ofstream os(GetKey() + "cuda.performance");
    os << result;
    os.close();
    return result;
  }

  template<bool print = true>
  OpenMPPerformance CheckPerformance(std::function<SerialParallelComparison()> task) {
    unsigned long long best_threads = 1;
    double best_speedup = 1;
    unsigned long long max_avalaible_threads = omp_get_max_threads();
    TEST_COUT << "Max avalaible threads: " << max_avalaible_threads << '\n';
    for(unsigned long long i = 2; i <= max_avalaible_threads; i++) {
      TEST_COUT << "Iteration for: " << i << " threads." << '\n';
      omp_set_num_threads(i);
      SerialParallelComparison comparison = task();
      if(print) {
        PRINT_SERIAL_PARALLEL_COMPARISON(comparison);
      }
      best_threads = comparison.GetSpeedup() > best_speedup ? i : best_threads;
      best_speedup = comparison.GetSpeedup() > best_speedup ? comparison.GetSpeedup() : best_speedup;
    }

    return  OpenMPPerformance(best_threads, best_speedup);
  }

  DoubleFloatComparison FloatDoubleCRSComparison(sspp::common::AbstractCRSSolver<float> & solver_float,
                                                 sspp::common::AbstractCRSSolver<double> & solver_double,
                                                 unsigned long long iterations) {
    auto crs_float = this->GetCRS<float>();
    auto vector_float = this->GetRandomVector<float>(crs_float.GetRows());
    auto crs_double = this->GetCRS<double>();
    auto vector_double = this->GetRandomVector<double>(crs_double.GetRows());

    double float_time, double_time;
    sspp::common::Output<float> output_float;
    sspp::common::Output<double> output_double;

    float_time = PerformIterations(
      [&solver_float, &crs_float, &vector_float, &output_float]() {
      output_float = solver_float.Solve(crs_float, vector_float);
      return output_float.GetSeconds();
    }, iterations);

    double_time = PerformIterations(
      [&solver_double, &crs_double, &vector_double, &output_double]() {
      output_double = solver_double.Solve(crs_double, vector_double);
      return output_double.GetSeconds();
    }, iterations);

    auto float_values = output_float.GetValues();
    auto double_values = output_double.GetValues();
    double delta = GetNormOfVectors(float_values, double_values);
    unsigned long long non_zeros_factor = 2 * crs_float.GetNonZeros();

    return DoubleFloatComparison(float_time, double_time, non_zeros_factor / float_time, non_zeros_factor / double_time, delta);
  }

  DoubleFloatComparison FloatDoubleELLPACKComparison(sspp::common::AbstractELLPACKSolver<float> & solver_float,
                                                     sspp::common::AbstractELLPACKSolver<double> & solver_double,
                                                     unsigned long long iterations) {
    auto ellpack_float = this->GetELLPACK<float>();
    auto vector_float = this->GetRandomVector<float>(ellpack_float.GetRows());
    auto ellpack_double = this->GetELLPACK<double>();
    auto vector_double = this->GetRandomVector<double>(ellpack_double.GetRows());

    double float_time, double_time;
    sspp::common::Output<float> output_float;
    sspp::common::Output<double> output_double;

    float_time = PerformIterations(
      [&solver_float, &ellpack_float, &vector_float, &output_float]() {
      output_float = solver_float.Solve(ellpack_float, vector_float);
      return output_float.GetSeconds();
    }, iterations);

    double_time = PerformIterations(
      [&solver_double, &ellpack_double, &vector_double, &output_double]() {
      output_double = solver_double.Solve(ellpack_double, vector_double);
      return output_double.GetSeconds();
    }, iterations);

    auto float_values = output_float.GetValues();
    auto double_values = output_double.GetValues();
    double delta = GetNormOfVectors(float_values, double_values);
    unsigned long long non_zeros_factor = 2 * ellpack_float.GetNonZeros();

    return DoubleFloatComparison(float_time, double_time, non_zeros_factor / float_time, non_zeros_factor / double_time, delta);
  }

  template<typename T>
  SerialParallelComparison SpeedupCRS(sspp::common::AbstractCRSSolver<T> & serial_solver,
                                      sspp::common::AbstractCRSSolver<T> & parallel_solver,
                                      unsigned long long iterations) {
    auto crs = this->GetCRS<T>();
    auto vector = this->GetRandomVector<T>(crs.GetRows());

    double serial_time, parallel_time;
    sspp::common::Output<T> serial_output, parallel_output;

    serial_time = PerformIterations(
      [&serial_solver, &crs, &vector, &serial_output]() {
      serial_output = serial_solver.Solve(crs, vector);
      return serial_output.GetSeconds();
    }, iterations);

    parallel_time = PerformIterations(
      [&parallel_solver, &crs, &vector, &parallel_output]() {
      parallel_output = parallel_solver.Solve(crs, vector);
      return parallel_output.GetSeconds();
    }, iterations);

    double delta = GetNormOfVectors(serial_output.GetValues(), parallel_output.GetValues());
    unsigned long long non_zeros_factor = 2 * crs.GetNonZeros();
    double speedup = serial_time / parallel_time;
    return SerialParallelComparison(serial_time, non_zeros_factor / serial_time,
                                    parallel_time, non_zeros_factor / parallel_time,
                                    speedup,
                                    delta);
  }

  template<typename T>
  SerialParallelComparison SpeedupEllpack(sspp::common::AbstractELLPACKSolver<T> & serial_solver,
                                          sspp::common::AbstractELLPACKSolver<T> & parallel_solver,
                                          unsigned long long iterations) {
    auto ellpack = this->GetELLPACK<T>();
    auto vector = this->GetRandomVector<T>(ellpack.GetRows());

    double serial_time, parallel_time;
    sspp::common::Output<T> serial_output, parallel_output;

    serial_time = PerformIterations(
      [&serial_solver, &ellpack, &vector, &serial_output]() {
      serial_output = serial_solver.Solve(ellpack, vector);
      return serial_output.GetSeconds();
    }, iterations);

    parallel_time = PerformIterations(
      [&parallel_solver, &ellpack, &vector, &parallel_output]() {
      parallel_output = parallel_solver.Solve(ellpack, vector);
      return parallel_output.GetSeconds();
    }, iterations);

    double delta = GetNormOfVectors(serial_output.GetValues(), parallel_output.GetValues());
    unsigned long long non_zeros_factor = 2 * ellpack.GetNonZeros();
    double speedup = serial_time / parallel_time;
    return SerialParallelComparison(serial_time, non_zeros_factor / serial_time,
                                    parallel_time, non_zeros_factor / parallel_time,
                                    speedup,
                                    delta);
  }

  template<typename T>
  std::vector<T> GetRandomVector(unsigned long long size, unsigned long long seed = 0) {
    std::vector<T> vector(size);
    srand(seed);
    for(unsigned long long i = 0; i < size; i++) {
      vector[i] = static_cast<T>(rand() % 100);
    }
    return vector;
  }

  template<typename T>
  sspp::common::CRS<T> GetCRS() {
    TEST_COUT << "Transforming to CRS<" + std::string(typeid(T).name()) + ">: " << GetKey();
    stopwatch_.Start();
    auto crs = MatrixContainer::GetInstance().GetCRS<T>(GetKey());
    stopwatch_.Stop();
    TEST_COUT_APPEND << " [" << stopwatch_.GetElapsedSeconds() << "s]" << std::endl;
    TEST_COUT << "Nonzeros: " << crs.GetNonZeros() << std::endl;
    return crs;
  }

  template<typename T>
  sspp::common::ELLPACK<T> GetELLPACK() {
    TEST_COUT << "Transforming to ELLPACK<" + std::string(typeid(T).name()) + ">: " << GetKey();
    stopwatch_.Start();
    auto ellpack = MatrixContainer::GetInstance().GetELLPACK<T>(GetKey());
    stopwatch_.Stop();
    TEST_COUT_APPEND << " [" << stopwatch_.GetElapsedSeconds() << "s]" << std::endl;
    TEST_COUT << "Nonzeros: " << ellpack.GetNonZeros() << std::endl;
    TEST_COUT << "Max row nonzeros: " << ellpack.GetMaxRowNonZeros() << std::endl;
    return ellpack;
  }

  double PerformIterations(std::function<double(void)> task, unsigned long long n) {
    double result = 0.0;
    for(unsigned long long i = 0; i < n; ++i) {
      result += task();
    }
    return result;
  }

  template<typename LESS_PRECISE, typename MORE_PRECISE>
  double GetNormOfVectors(std::vector<LESS_PRECISE> & lhs, std::vector<MORE_PRECISE> & rhs) {
    double delta = 0.0;
    for(unsigned long long i = 0; i < lhs.size(); i++) {
      delta += fabs(static_cast<MORE_PRECISE>(lhs[i]) - rhs[i]);
    }
    return delta;
  }

  double GetSpeedUpFor(std::function<double(void)> reference, std::function<double(void)> actual, unsigned long long n) {
    return GetSpeedUp(PerformIterations(reference, n), PerformIterations(actual, n));
  }

  double GetSpeedUp(double reference, double actual) {
    return reference / actual;
  }

  virtual std::string GetPath() = 0;
  virtual std::string GetKey() = 0;

  sspp::common::ChronoStopwatch stopwatch_;
  static std::string static_key_;
};

std::string MatrixTestInterface::static_key_ = "";
