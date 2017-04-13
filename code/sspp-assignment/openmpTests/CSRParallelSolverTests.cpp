#include "CRSParallelSolverTest.h"
#include <gtest\gtest.h>
#include <gmock/gmock.h>

TEST_F(CRSParallelSolverTest, test) {
  const unsigned long long M = 4, N = 4, NZ = 7, THREADS = 2;
  std::vector<unsigned long long> IRP = { 0, 2, 4, 5, 7 },
    JA = { 0, 1, 1, 2, 2, 2, 3 };
  std::vector<float> AS = { 11, 12, 22, 23, 33, 43, 44 };
  sspp::common::CRS<float> csr(M, N, NZ, IRP, JA, AS);
  std::vector<float> B = { 1, 1, 1, 1 };
  std::vector<unsigned long long> correctX = { 23, 45, 33, 87 };

  csrParallelSolver->SetThreads(THREADS);
  auto output = csrParallelSolver->Solve(csr, B);

  ASSERT_EQ(M, output.GetValues().size());
  ASSERT_THAT(output.GetValues(), ::testing::ElementsAre(23, 45, 33, 87));
}