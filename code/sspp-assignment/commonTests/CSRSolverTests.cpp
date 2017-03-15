#include "CSRSolverTest.h"
#include "Definitions.h"
#include "CSR.h"
#include <gtest\gtest.h>

TEST_F(CSRSolverTest, shouldSolveCorrectly_Salvatore) {
  const int M = 4, N = 4, NZ = 7;
  int *IRP = new int[NZ], *JA = new int[NZ];
  FLOATING_TYPE *AS = new FLOATING_TYPE[NZ];

  int correctIRP[5] = { 0, 2, 4, 5, 7 };
  for(int i = 0; i < 5; i++)
    IRP[i] = correctIRP[i];
  int correctJA[NZ] = { 0, 1, 1, 2, 2, 2, 3 };
  for(int i = 0; i < NZ; i++)
    JA[i] = correctJA[i];
  FLOATING_TYPE correctAS[NZ] = { 11, 12, 22, 23, 33, 43, 44 };
  for(int i = 0; i < NZ; i++)
    AS[i] = correctAS[i];
  sspp::representations::CSR csr(NZ, M, N, IRP, JA, AS);

  FLOATING_TYPE B[N] = { 1, 1, 1, 1 };
  FLOATING_TYPE correctX[M] = { 23, 45, 33, 87 };

  auto output = csrSolver->solve(csr, B);

  ASSERT_EQ(M, output.N) << "Size of output is incorrect";

  assertArrays(correctX, output.Values, M, "X -> Incorrect value at: ");

}

TEST_F(CSRSolverTest, shouldSolveCorrectly) {
  const int M = 3, N = 4, NZ = 4;
  int *IRP = new int[NZ], *JA = new int[NZ];
  FLOATING_TYPE *AS = new FLOATING_TYPE[NZ];

  int correctIRP[4] = { 0, 1, 2, 4 };
  for(int i = 0; i < 4; i++)
    IRP[i] = correctIRP[i];
  int correctJA[NZ] = { 0, 1, 2, 3 };
  for(int i = 0; i < NZ; i++)
    JA[i] = correctJA[i];
  FLOATING_TYPE correctAS[NZ] = { 15, 20, 1, 5 };
  for(int i = 0; i < NZ; i++)
    AS[i] = correctAS[i];
  sspp::representations::CSR csr(NZ, M, N, IRP, JA, AS);

  FLOATING_TYPE B[N] = { 2, 1, 3, 4 };
  FLOATING_TYPE correctX[M] = { 30, 20, 23 };

  auto output = csrSolver->solve(csr, B);

  ASSERT_EQ(M, output.N) << "Size of output is incorrect";

  assertArrays(correctX, output.Values, M, "X -> Incorrect value at: ");

}