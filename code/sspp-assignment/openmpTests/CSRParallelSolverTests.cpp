#include "CSRParallelSolverTest.h"
#include <gtest\gtest.h>
#include "../openmp/CSROpenMPSolver.h"

TEST_F(CSRParallelSolverTest, test) {
  const INDEXING_TYPE M = 4, N = 4, NZ = 7, THREADS = 2;
  auto IRP = new std::vector<INDEXING_TYPE>{ 0, 2, 4, 5, 7 },
    JA = new std::vector<INDEXING_TYPE>{ 0, 1, 1, 2, 2, 2, 3 };
  auto AS = new std::vector<FLOATING_TYPE>{ 11, 12, 22, 23, 33, 43, 44 };
  sspp::representations::CSR csr(NZ, M, N, IRP, JA, AS);
  std::vector<FLOATING_TYPE> B = { 1, 1, 1, 1 };
  std::vector<FLOATING_TYPE> correctX = { 23, 45, 33, 87 };

  csrParallelSolver->setThreads(THREADS);
  auto output = csrParallelSolver->Solve(csr, B);

  ASSERT_EQ(M, output->GetValues()->size()) << "Size of output_ is incorrect";
  assertArrays(correctX, *output->GetValues(), M, "X -> Incorrect value at: ");
}