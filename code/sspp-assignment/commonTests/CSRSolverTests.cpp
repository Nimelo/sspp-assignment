#include "CSRSolverTest.h"

#include "CRS.h"
#include "CRSSolver.h"
#include <gtest\gtest.h>
#include <gmock/gmock.h>
#include <vector>

using namespace sspp::common;

TEST_F(CSRSolverTest, shouldSolveCorrectly_Salvatore) {
  const unsigned M = 4, N = 4, NZ = 7;
  std::vector<unsigned> IRP = { 0, 2, 4, 5, 7 },
    JA = { 0, 1, 1, 2, 2, 2, 3 };
  std::vector<float> AS = { 11, 12, 22, 23, 33, 43, 44 };
  CRS<float> csr(M, N, NZ, IRP, JA, AS);
  std::vector<float> B = { 1, 1, 1, 1 };
  std::vector<float> correctX = { 23, 45, 33, 87 };
  CRSSolver<float> solver;

  const Output<float> &output = solver.Solve(csr, B);

  ASSERT_EQ(M, output.GetValues().size());
  ASSERT_THAT(correctX, ::testing::ContainerEq(output.GetValues()));
}

TEST_F(CSRSolverTest, shouldSolveCorrectly) {
  const unsigned M = 3, N = 4, NZ = 4;
  std::vector<unsigned> IRP = { 0, 1, 2, 4 },
    JA = { 0, 1, 2, 3 };
  std::vector<float> AS = { 15, 20, 1, 5 };
  CRS<float> csr(M, N, NZ, IRP, JA, AS);
  std::vector<float> B = { 2, 1, 3, 4 };
  std::vector<float> correctX = { 30, 20, 23 };

  CRSSolver<float> solver;
  const Output<float> output = solver.Solve(csr, B);

  ASSERT_EQ(M, output.GetValues().size());
  ASSERT_THAT(correctX, ::testing::ContainerEq(output.GetValues()));
}