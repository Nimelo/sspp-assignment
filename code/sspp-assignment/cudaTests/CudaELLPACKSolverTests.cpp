#include "CudaELLPACKSolverTest.h"
#include <gtest\gtest.h>
#include "../openmp/ELLPACKOpenMPSolver.h"

TEST_F(CudaELLPACKSolverTest, test) {
  const int M = 4, N = 4, NZ = 7, MAXNZ = 2, THREADS = 2;
  int **JA = new int*[M];
  FLOATING_TYPE **AS = new FLOATING_TYPE*[M];

  FLOATING_TYPE correctAS[M][MAXNZ] = { { 11, 12 },{ 22, 23 },{ 33, 0 },{ 43, 44 } };
  int correctJA[M][MAXNZ] = { { 0, 1 },{ 1, 2 },{ 2, 2 },{ 2, 3 } };

  for(int i = 0; i < M; i++) {
    JA[i] = new int[MAXNZ];
    AS[i] = new FLOATING_TYPE[MAXNZ];
    for(int j = 0; j < MAXNZ; j++) {
      JA[i][j] = correctJA[i][j];
      AS[i][j] = correctAS[i][j];
    }
  }

  sspp::representations::ELLPACK ellpack(M, N, NZ, MAXNZ, JA, AS);

  FLOATING_TYPE B[N] = { 1, 1, 1, 1 };
  FLOATING_TYPE correctX[M] = { 23, 45, 33, 87 };

  auto output = this->csrParallelSolver->solve(ellpack, B);

  ASSERT_EQ(M, output.N) << "Size of output is incorrect";

  assertArrays(correctX, output.Values, M, "X -> Incorrect value at: ");
}