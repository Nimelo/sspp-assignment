#include "ELLPACKTransformerTest.h"
#include "Definitions.h"
#include <gtest\gtest.h>
#include <sstream>

TEST_F(ELLPACKTransformerTest, shouldSolveCorrectly_Salvatore) {
  const int M = 4, N = 4, NZ = 7, correctMAXNZ = 2;
  int *iIndexes = new int[NZ], *jIndexes = new int[NZ];
  FLOATING_TYPE *values = new FLOATING_TYPE[NZ];
  iIndexes[0] = 0; iIndexes[1] = 0; iIndexes[2] = 1; iIndexes[3] = 1; iIndexes[4] = 2; iIndexes[5] = 3; iIndexes[6] = 3;
  jIndexes[0] = 0; jIndexes[1] = 1; jIndexes[2] = 1; jIndexes[3] = 2; jIndexes[4] = 2; jIndexes[5] = 2; jIndexes[6] = 3;
  values[0] = 11; values[1] = 12; values[2] = 22; values[3] = 23; values[4] = 33; values[5] = 43; values[6] = 44;

  FLOATING_TYPE correctAS[M][correctMAXNZ] = { { 11, 12 },{ 22, 23 },{ 33, 0 },{ 43, 44 } };
  int correctJA[M][correctMAXNZ] = { { 0, 1 },{ 1, 2 },{ 2, 2 },{ 2, 3 } };

  sspp::representations::IntermediarySparseMatrix ism(M, N, NZ, iIndexes, jIndexes, values);
  auto ellpack = ellpackTransformer->transform(ism);

  ASSERT_EQ(M, ellpack.M) << "M values is different.";
  ASSERT_EQ(N, ellpack.N) << "N values is different.";
  ASSERT_EQ(NZ, ellpack.NZ) << "NZ values is different.";
  ASSERT_EQ(correctMAXNZ, ellpack.MAXNZ) << "MAXNZ values is different.";


  for(int i = 0; i < M; i++) {
    for(int j = 0; j < correctMAXNZ; j++) {
      ASSERT_EQ(correctJA[i][j], ellpack.JA[i][j]) << "JA values is different.";
      ASSERT_EQ(correctAS[i][j], ellpack.AS[i][j]) << "AS values is different.";
    }
  }
}


TEST_F(ELLPACKTransformerTest, shouldSolveCorrectly) {
  const int M = 3, N = 4, NZ = 5, correctMAXNZ = 3;
  int *iIndexes = new int[NZ], *jIndexes = new int[NZ];
  FLOATING_TYPE *values = new FLOATING_TYPE[NZ];
  iIndexes[0] = 0; iIndexes[1] = 0; iIndexes[2] = 0; iIndexes[3] = 1; iIndexes[4] = 2;
  jIndexes[0] = 0; jIndexes[1] = 1; jIndexes[2] = 2; jIndexes[3] = 2; jIndexes[4] = 3;
  values[0] = 2; values[1] = 7; values[2] = 1; values[3] = 4; values[4] = 1;

  FLOATING_TYPE correctAS[M][correctMAXNZ] = { { 2, 7, 1 },{ 4, 0, 0 },{ 1, 0, 0 } };
  int correctJA[M][correctMAXNZ] = { { 0, 1, 2 },{ 2, 2, 2 },{ 3, 3, 3 } };

  sspp::representations::IntermediarySparseMatrix ism(M, N, NZ, iIndexes, jIndexes, values);
  auto ellpack = ellpackTransformer->transform(ism);

  ASSERT_EQ(M, ellpack.M) << "M values is different.";
  ASSERT_EQ(N, ellpack.N) << "N values is different.";
  ASSERT_EQ(NZ, ellpack.NZ) << "NZ values is different.";
  ASSERT_EQ(correctMAXNZ, ellpack.MAXNZ) << "MAXNZ values is different.";


  for(int i = 0; i < M; i++) {
    for(int j = 0; j < correctMAXNZ; j++) {
      ASSERT_EQ(correctJA[i][j], ellpack.JA[i][j]) << "JA values is different.";
      ASSERT_EQ(correctAS[i][j], ellpack.AS[i][j]) << "AS values is different.";
    }
  }
}

TEST_F(ELLPACKTransformerTest, iostreamTest) {
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

  std::stringstream stringStream;

  stringStream << ellpack;

  sspp::representations::ELLPACK actualEllpack;

  stringStream >> actualEllpack;

  ASSERT_EQ(M, ellpack.M) << "M values is different.";
  ASSERT_EQ(N, ellpack.N) << "N values is different.";
  ASSERT_EQ(NZ, ellpack.NZ) << "NZ values is different.";
  ASSERT_EQ(MAXNZ, ellpack.MAXNZ) << "MAXNZ values is different.";


  for(int i = 0; i < M; i++) {
    for(int j = 0; j < MAXNZ; j++) {
      ASSERT_EQ(ellpack.JA[i][j], actualEllpack.JA[i][j]) << "JA values is different.";
      ASSERT_EQ(ellpack.AS[i][j], actualEllpack.AS[i][j]) << "AS values is different.";
    }
  }
}