#include "CSRTransformerTest.h"
#include "Definitions.h"
#include <gtest\gtest.h>
#include <sstream>
#include <string>

TEST_F(CSRTransformerTest, shouldTransformCorrectly_Salvatore) {
  const int M = 4, N = 4, NZ = 7;
  int *iIndexes = new int[NZ], *jIndexes = new int[NZ];
  FLOATING_TYPE *values = new FLOATING_TYPE[NZ];
  iIndexes[0] = 0; iIndexes[1] = 0; iIndexes[2] = 1; iIndexes[3] = 1; iIndexes[4] = 2; iIndexes[5] = 3; iIndexes[6] = 3;
  jIndexes[0] = 0; jIndexes[1] = 1; jIndexes[2] = 1; jIndexes[3] = 2; jIndexes[4] = 2; jIndexes[5] = 2; jIndexes[6] = 3;
  values[0] = 11; values[1] = 12; values[2] = 22; values[3] = 23; values[4] = 33; values[5] = 43; values[6] = 44;
  int correctIRP[5] = { 0, 2, 4, 5, 7 };
  int correctJA[NZ] = { 0, 1, 1, 2, 2, 2, 3 };

  sspp::representations::IntermediarySparseMatrix ism(M, N, NZ, iIndexes, jIndexes, values);
  auto csr = csrTransformer->transform(ism);

  ASSERT_EQ(M, csr.M) << "M values is different.";
  ASSERT_EQ(N, csr.N) << "N values is different.";
  ASSERT_EQ(NZ, csr.getJASize()) << "JASize values is different.";
  ASSERT_EQ(NZ, csr.getASSize()) << "ASSize values is different.";
  ASSERT_EQ(M + 1, csr.getIRPSize()) << "IRPSize values is different.";

  assertArrays(values, csr.AS, NZ, "AS -> Incorrect value at: ");
  assertArrays(correctIRP, csr.IRP, 5, "IRP -> Incorrect value at: ");
  assertArrays(correctJA, csr.JA, NZ, "JA -> Incorrect value at: ");
}

TEST_F(CSRTransformerTest, shouldTransformCorrectly) {
  const int M = 3, N = 4, NZ = 5;
  int *iIndexes = new int[NZ], *jIndexes = new int[NZ];
  FLOATING_TYPE *values = new FLOATING_TYPE[NZ];
  iIndexes[0] = 0; iIndexes[1] = 1; iIndexes[2] = 1; iIndexes[3] = 2; iIndexes[4] = 2;
  jIndexes[0] = 2; jIndexes[1] = 2; jIndexes[2] = 3; jIndexes[3] = 0; jIndexes[4] = 1;
  values[0] = 1; values[1] = 2; values[2] = 3; values[3] = 4; values[4] = 1;
  int correctIRP[4] = { 0, 1, 3, 5 };
  int correctJA[NZ] = { 2, 2, 3, 0, 1 };

  sspp::representations::IntermediarySparseMatrix ism(M, N, NZ, iIndexes, jIndexes, values);
  auto csr = csrTransformer->transform(ism);

  ASSERT_EQ(M, csr.M) << "M values is different.";
  ASSERT_EQ(N, csr.N) << "N values is different.";
  ASSERT_EQ(NZ, csr.getJASize()) << "JASize values is different.";
  ASSERT_EQ(NZ, csr.getASSize()) << "ASSize values is different.";
  ASSERT_EQ(M + 1, csr.getIRPSize()) << "IRPSize values is different.";

  assertArrays(values, csr.AS, NZ, "AS -> Incorrect value at: ");
  assertArrays(correctIRP, csr.IRP, 4, "IRP -> Incorrect value at: ");
  assertArrays(correctJA, csr.JA, NZ, "JA -> Incorrect value at: ");
}

TEST_F(CSRTransformerTest, iostreamTest) {
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
  sspp::representations::CSR expectedCSR(NZ, M, N, IRP, JA, AS);

  std::stringstream stringStream;
  stringStream << expectedCSR;

  sspp::representations::CSR actualCSR;

  stringStream >> actualCSR;

  ASSERT_EQ(expectedCSR.M, actualCSR.M);
  ASSERT_EQ(expectedCSR.N, actualCSR.N);
  ASSERT_EQ(expectedCSR.NZ, actualCSR.NZ);

  assertArrays(expectedCSR.IRP, actualCSR.IRP, expectedCSR.getIRPSize(), "Incorrect IRP at: ");
  assertArrays(expectedCSR.JA, actualCSR.JA, expectedCSR.getIRPSize(), "Incorrect JA at: ");
  assertArrays(expectedCSR.AS, actualCSR.AS, expectedCSR.getIRPSize(), "Incorrect AS at: ");
}