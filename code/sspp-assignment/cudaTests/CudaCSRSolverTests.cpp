#include "CudaCSRSolverTest.h"
#include <gtest\gtest.h>

TEST_F(CudaCSRSolverTest, shouldSolveCorrectly_Salvatore) {
  const INDEXING_TYPE M = 4, N = 4, NZ = 7, THREADS = 2;
  auto IRP = new std::vector<INDEXING_TYPE>{ 0, 2, 4, 5, 7 },
    JA = new std::vector<INDEXING_TYPE>{ 0, 1, 1, 2, 2, 2, 3 };
  auto AS = new std::vector<FLOATING_TYPE>{ 11, 12, 22, 23, 33, 43, 44 };
  sspp::representations::CSR csr(NZ, M, N, IRP, JA, AS);
  std::vector<FLOATING_TYPE> B = { 1, 1, 1, 1 };
  std::vector<FLOATING_TYPE> correctX = { 23, 45, 33, 87 };

  auto output = csrParallelSolver->Solve(csr, B);

  ASSERT_EQ(M, output->GetValues()->size()) << "Size of output_ is incorrect";
  assertArrays(correctX, *output->GetValues(), M, "X -> Incorrect value at: ");
  delete output;
}

TEST_F(CudaCSRSolverTest, shouldSolveCorrectly) {

  const INDEXING_TYPE M = 3, N = 4, NZ = 4;
  auto IRP = new std::vector<INDEXING_TYPE>{ 0, 1, 2, 4 },
    JA = new std::vector<INDEXING_TYPE>{ 0, 1, 2, 3 };
  auto AS = new std::vector<FLOATING_TYPE>{ 15, 20, 1, 5 };
  sspp::representations::CSR csr(NZ, M, N, IRP, JA, AS);
  std::vector<FLOATING_TYPE> B = { 2, 1, 3, 4 };
  std::vector<FLOATING_TYPE> correctX = { 30, 20, 23 };

  auto output = csrParallelSolver->Solve(csr, B);

  ASSERT_EQ(M, output->GetValues()->size()) << "Size of output_ is incorrect";
  assertArrays(correctX, *output->GetValues(), M, "X -> Incorrect value at: ");
  delete output;
}