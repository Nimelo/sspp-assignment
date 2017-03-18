#include "CSRSolverTest.h"
#include "Definitions.h"
#include "CSR.h"
#include <gtest\gtest.h>
#include <vector>

TEST_F(CSRSolverTest, shouldSolveCorrectly_Salvatore) {
  const INDEXING_TYPE M = 4, N = 4, NZ = 7;
  std::vector<INDEXING_TYPE> IRP = { 0, 2, 4, 5, 7 },
    JA = { 0, 1, 1, 2, 2, 2, 3 };
  std::vector<FLOATING_TYPE> AS = { 11, 12, 22, 23, 33, 43, 44 };
  sspp::representations::CSR csr(NZ, M, N, IRP, JA, AS);
  std::vector<FLOATING_TYPE> B = { 1, 1, 1, 1 };
  FLOATING_TYPE correctX[M] = { 23, 45, 33, 87 };

  auto output = csrSolver->Solve(csr, B);

  ASSERT_EQ(M, output.GetValues().size()) << "Size of output_ is incorrect";
  assertArrays(correctX, &output.GetValues()[0], M, "X -> Incorrect value at: ");
}

TEST_F(CSRSolverTest, shouldSolveCorrectly) {

  const INDEXING_TYPE M = 3, N = 4, NZ = 4;
  std::vector<INDEXING_TYPE> IRP = { 0, 1, 2, 4 },
    JA = { 0, 1, 2, 3 };
  std::vector<FLOATING_TYPE> AS = { 15, 20, 1, 5 };
  sspp::representations::CSR csr(NZ, M, N, IRP, JA, AS);
  std::vector<FLOATING_TYPE> B = { 2, 1, 3, 4 };
  FLOATING_TYPE correctX[M] = { 30, 20, 23 };

  auto output = csrSolver->Solve(csr, B);

  ASSERT_EQ(M, output.GetValues().size()) << "Size of output_ is incorrect";
  assertArrays(correctX, &output.GetValues()[0], M, "X -> Incorrect value at: ");
}