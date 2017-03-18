#include "CudaELLPACKSolverTest.h"
#include <gtest\gtest.h>
#include "../openmp/ELLPACKOpenMPSolver.h"

TEST_F(CudaELLPACKSolverTest, test) {
  const INDEXING_TYPE M = 4, N = 4, NZ = 7, MAXNZ = 2, THREADS = 2;
  std::vector<INDEXING_TYPE> JA = { 0, 1, 1, 2, 2, 2, 2, 3 };
  std::vector<FLOATING_TYPE> AS = { 11, 12, 22, 23, 33, 0, 43, 44 };
  sspp::representations::ELLPACK ellpack(M, N, NZ, MAXNZ, JA, AS);
  std::vector<FLOATING_TYPE> B = { 1, 1, 1, 1 };
  std::vector<FLOATING_TYPE> correctX = { 23, 45, 33, 87 };

  auto output = this->csrParallelSolver->Solve(ellpack, B);

  ASSERT_EQ(M, output.GetValues().size()) << "Size of output_ is incorrect";
  assertArrays(&correctX[0], &output.GetValues()[0], M, "X -> Incorrect value at: ");
}