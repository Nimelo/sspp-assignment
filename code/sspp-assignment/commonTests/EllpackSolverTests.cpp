#include "EllpackSolverTest.h"
#include "Definitions.h"
#include "ELLPACK.h"
#include <gtest\gtest.h>
#include <vector>

TEST_F(ELLPACKSolverTest, shouldSolveCorrectly_Salvatore) {
  const INDEXING_TYPE M = 4, N = 4, NZ = 7, MAXNZ = 2;
  auto JA = new std::vector<INDEXING_TYPE>{ 0, 1 ,1, 2 , 2, 2 , 2, 3 };
  auto AS = new std::vector<FLOATING_TYPE>{ 11, 12 , 22, 23 , 33, 0 , 43, 44 };
  sspp::representations::ELLPACK ellpack(M, N, NZ, MAXNZ, JA, AS);
  std::vector<FLOATING_TYPE> B = { 1, 1, 1, 1 };
  FLOATING_TYPE correctX[M] = { 23, 45, 33, 87 };

  auto output = ellpackSolver->Solve(ellpack, B);

  ASSERT_EQ(M, output->GetValues()->size()) << "Size of output_ is incorrect";
  assertArrays(correctX, &(*output->GetValues())[0], M, "X -> Incorrect value at: ");
  delete output;
}

TEST_F(ELLPACKSolverTest, shouldSolveCorrectly) {
  const INDEXING_TYPE M = 3, N = 4, NZ = 5, MAXNZ = 3;
  auto JA = new std::vector<INDEXING_TYPE>{ 0, 1, 2 , 2, 2, 2 , 3, 3, 3 };
  auto AS = new std::vector<FLOATING_TYPE>{ 2, 7, 1 , 4, 0, 0 , 1, 0, 0 };
  sspp::representations::ELLPACK ellpack(M, N, NZ, MAXNZ, JA, AS);
  std::vector<FLOATING_TYPE> B = { 1, 1, 1, 1 };
  FLOATING_TYPE correctX[M] = { 10, 4, 1 };

  auto output = ellpackSolver->Solve(ellpack, B);

  ASSERT_EQ(M, output->GetValues()->size()) << "Size of output_ is incorrect";
  assertArrays(correctX, &(*output->GetValues())[0], M, "X -> Incorrect value at: ");
  delete output;
}
