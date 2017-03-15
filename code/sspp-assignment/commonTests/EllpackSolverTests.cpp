#include "EllpackSolverTest.h"
#include "Definitions.h"
#include "ELLPACK.h"
#include <gtest\gtest.h>

TEST_F(ELLPACKSolverTest, shouldSolveCorrectly_Salvatore) {
  const int M = 4, N = 4, NZ = 7, MAXNZ = 2;
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

  auto output = ellpackSolver->solve(ellpack, B);

  ASSERT_EQ(M, output.N) << "Size of output is incorrect";

  assertArrays(correctX, output.Values, M, "X -> Incorrect value at: ");

}

TEST_F(ELLPACKSolverTest, shouldSolveCorrectly) {
  const int M = 3, N = 4, NZ = 5, MAXNZ = 3;
  int **JA = new int*[M];
  FLOATING_TYPE **AS = new FLOATING_TYPE*[M];

  FLOATING_TYPE correctAS[M][MAXNZ] = { { 2, 7, 1 },{ 4, 0, 0 },{ 1, 0, 0 } };
  int correctJA[M][MAXNZ] = { { 0, 1, 2 },{ 2, 2, 2 },{ 3, 3, 3 } };

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
  FLOATING_TYPE correctX[M] = { 10, 4, 1 };

  auto output = ellpackSolver->solve(ellpack, B);

  ASSERT_EQ(M, output.N) << "Size of output is incorrect";

  assertArrays(correctX, output.Values, M, "X -> Incorrect value at: ");

}
