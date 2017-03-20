#include "CudaELLPACKSolverTest.h"
#include <gtest\gtest.h>
#include "../openmp/ELLPACKOpenMPSolver.h"

TEST_F(CudaELLPACKSolverTest, shouldSolveCorrectly_Salvatore) {
  const INDEXING_TYPE M = 4, N = 4, NZ = 7, MAXNZ = 2, THREADS = 2;
  auto JA = new std::vector<INDEXING_TYPE>{ 0, 1, 1, 2, 2, 2, 2, 3 };
  auto AS = new std::vector<FLOATING_TYPE>{ 11, 12, 22, 23, 33, 0, 43, 44 };
  sspp::representations::ELLPACK ellpack(M, N, NZ, MAXNZ, JA, AS);
  std::vector<FLOATING_TYPE> B = { 1, 1, 1, 1 };
  std::vector<FLOATING_TYPE> correctX = { 23, 45, 33, 87 };

  auto output = this->csrParallelSolver->Solve(ellpack, B);

  ASSERT_EQ(M, output->GetValues()->size()) << "Size of output_ is incorrect";
  assertArrays(correctX, *output->GetValues(), M, "X -> Incorrect value at: ");
  delete output;
}

TEST_F(CudaELLPACKSolverTest, shouldSolveCorrectly) {
  const INDEXING_TYPE M = 4, N = 4, NZ = 7, MAXNZ = 2;
  auto JA = new std::vector<INDEXING_TYPE>{ 0, 1 ,1, 2 , 2, 2 , 2, 3 };
  auto AS = new std::vector<FLOATING_TYPE>{ 11, 12 , 22, 23 , 33, 0 , 43, 44 };
  sspp::representations::ELLPACK ellpack(M, N, NZ, MAXNZ, JA, AS);
  std::vector<FLOATING_TYPE> B = { 1, 1, 1, 1 };
  std::vector<FLOATING_TYPE> correctX = { 23, 45, 33, 87 };

  auto output = csrParallelSolver->Solve(ellpack, B);

  ASSERT_EQ(M, output->GetValues()->size()) << "Size of output_ is incorrect";
  assertArrays(correctX, *output->GetValues(), M, "X -> Incorrect value at: ");
  delete output;
}