#include "EllpackSolverTest.h"

#include "ELLPACKSolver.h"
#include "ELLPACK.h"
#include <gtest\gtest.h>
#include <gmock/gmock.h>
#include <vector>

using namespace sspp::common;

TEST_F(ELLPACKSolverTest, shouldSolveCorrectly_Salvatore) {
  const unsigned long long M = 4, N = 4, NZ = 7, MAXNZ = 2;
  std::vector<unsigned long long> JA = { 0, 1 ,1, 2 , 2, 2 , 2, 3 };
  std::vector<float> AS = { 11, 12 , 22, 23 , 33, 0 , 43, 44 };
  ELLPACK<float> ellpack(M, N, NZ, MAXNZ, JA, AS);
  std::vector<float> B = { 1, 1, 1, 1 };
  std::vector<float> correctX = { 23, 45, 33, 87 };
  ELLPACKSolver<float> solver;

  const Output<float> output = solver.Solve(ellpack, B);

  ASSERT_EQ(M, output.GetValues().size());
  ASSERT_THAT(correctX, ::testing::ContainerEq(output.GetValues()));
}

TEST_F(ELLPACKSolverTest, shouldSolveCorrectly) {
  const unsigned long long M = 3, N = 4, NZ = 5, MAXNZ = 3;
  std::vector<unsigned long long> JA = { 0, 1, 2 , 2, 2, 2 , 3, 3, 3 };
  std::vector<float> AS = { 2, 7, 1 , 4, 0, 0 , 1, 0, 0 };
  ELLPACK<float> ellpack(M, N, NZ, MAXNZ, JA, AS);
  std::vector<float> B = { 1, 1, 1, 1 };
  std::vector<float> correctX = { 10, 4, 1 };
  ELLPACKSolver<float> solver;

  const Output<float> output = solver.Solve(ellpack, B);

  ASSERT_EQ(M, output.GetValues().size());
  ASSERT_THAT(correctX, ::testing::ContainerEq(output.GetValues()));
}
