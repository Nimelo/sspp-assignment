#include "ELLPACKParallelSolverTest.h"
#include <gtest\gtest.h>
#include "../openmp/ELLPACKOpenMPSolver.h"
#include <gmock/gmock.h>

TEST_F(ELLPACKParallelSolverTest, test) {
  const unsigned M = 4, N = 4, NZ = 7, MAXNZ = 2, THREADS = 2;
  std::vector<unsigned> JA = { 0, 1 , 1, 2 ,2, 2, 2, 3 };
  std::vector<float> AS = { 11, 12 , 22, 23, 33, 0, 43, 44 };
  sspp::common::ELLPACK<float> ellpack(M, N, NZ, MAXNZ, JA, AS);

  std::vector<float> B = { 1, 1, 1, 1 };
  std::vector<float> correctX = { 23, 45, 33, 87 };

  ellpackParallelSolver->SetThreads(THREADS);
  auto output = ellpackParallelSolver->Solve(ellpack, B);

  ASSERT_EQ(M, output.GetValues().size());
  ASSERT_THAT(output.GetValues(), ::testing::ElementsAre(23, 45, 33, 87));
}