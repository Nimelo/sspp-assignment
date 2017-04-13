#include "CudaELLPACKSolverTest.h"
#include <gtest\gtest.h>
#include <gmock/gmock.h>
#include "../openmp/ELLPACKOpenMPSolver.h"

TEST_F(CudaELLPACKSolverTest, shouldSolveCorrectly_Salvatore) {
  const unsigned long long M = 4, N = 4, NZ = 7, MAXNZ = 2, THREADS = 2;
  std::vector<unsigned long long> JA = { 0, 1, 1, 2, 2, 2, 2, 3 };
  std::vector<float> AS = { 11, 12, 22, 23, 33, 0, 43, 44 };
  sspp::common::ELLPACK<float> ellpack(M, N, NZ, MAXNZ, JA, AS);
  std::vector<float> B = { 1, 1, 1, 1 };
  std::vector<float> correctX = { 23, 45, 33, 87 };

  auto output = this->csrParallelSolver->Solve(ellpack, B);

  ASSERT_EQ(M, output.GetValues().size()) << "Size of output_ is incorrect";
  ASSERT_THAT(output.GetValues(), ::testing::ElementsAre(23, 45, 33, 87));
}

TEST_F(CudaELLPACKSolverTest, shouldSolveCorrectly) {
  const unsigned long long M = 4, N = 4, NZ = 7, MAXNZ = 2;
  std::vector<unsigned long long> JA = { 0, 1 ,1, 2 , 2, 2 , 2, 3 };
  std::vector<float> AS = { 11, 12 , 22, 23 , 33, 0 , 43, 44 };
  sspp::common::ELLPACK<float> ellpack(M, N, NZ, MAXNZ, JA, AS);
  std::vector<float> B = { 1, 1, 1, 1 };
  float correctX[M] = { 23, 45, 33, 87 };

  auto output = csrParallelSolver->Solve(ellpack, B);

  ASSERT_EQ(M, output.GetValues().size()) << "Size of output_ is incorrect";
  ASSERT_THAT(output.GetValues(), ::testing::ElementsAre(23, 45, 33, 87));
}