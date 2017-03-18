#include "CSRTransformerTest.h"
#include "Definitions.h"
#include <gtest\gtest.h>
#include <sstream>
#include <vector>
#include <string>

TEST_F(CSRTransformerTest, shouldTransformCorrectly_Salvatore) {
  const int M = 4, N = 4, NZ = 7;
  std::vector<INDEXING_TYPE> iIndexes = { 0, 0, 1, 1, 2, 3, 3 },
    jIndexes = { 0, 1, 1, 2, 2, 2, 3 };
  std::vector<FLOATING_TYPE> values = { 11, 12, 22, 23, 33, 43, 44 };

  INDEXING_TYPE correctIRP[5] = { 0, 2, 4, 5, 7 },
    correctJA[NZ] = { 0, 1, 1, 2, 2, 2, 3 };

  sspp::representations::IntermediarySparseMatrix ism(M, N, NZ, iIndexes, jIndexes, values);
  auto csr = csrTransformer->Transform(ism);

  ASSERT_EQ(M, csr.GetRows()) << "rows_ values is different.";
  ASSERT_EQ(N, csr.GetColumns()) << "columns_ values is different.";
  ASSERT_EQ(NZ, csr.GetJA().size()) << "JASize values is different.";
  ASSERT_EQ(NZ, csr.GetAS().size()) << "ASSize values is different.";
  ASSERT_EQ(M + 1, csr.GetIRP().size()) << "IRPSize values is different.";

  assertArrays(values, csr.GetAS(), NZ, "as_ -> Incorrect value at: ");
  assertArrays(correctIRP, &csr.GetIRP()[0], 5, "irp_ -> Incorrect value at: ");
  assertArrays(correctJA, &csr.GetJA()[0], NZ, "ja_ -> Incorrect value at: ");
}

TEST_F(CSRTransformerTest, shouldTransformCorrectly) {
  const int M = 3, N = 4, NZ = 5;
  std::vector<INDEXING_TYPE> iIndexes = { 0, 1, 1, 2, 2 },
    jIndexes = { 2, 2, 3, 0, 1 };
  std::vector<FLOATING_TYPE> values = { 1, 2, 3, 4, 1 };
  INDEXING_TYPE correctIRP[4] = { 0, 1, 3, 5 },
    correctJA[NZ] = { 2, 2, 3, 0, 1 };

  sspp::representations::IntermediarySparseMatrix ism(M, N, NZ, iIndexes, jIndexes, values);
  auto csr = csrTransformer->Transform(ism);

  ASSERT_EQ(M, csr.GetRows()) << "rows_ values is different.";
  ASSERT_EQ(N, csr.GetColumns()) << "columns_ values is different.";
  ASSERT_EQ(NZ, csr.GetJA().size()) << "JASize values is different.";
  ASSERT_EQ(NZ, csr.GetAS().size()) << "ASSize values is different.";
  ASSERT_EQ(M + 1, csr.GetIRP().size()) << "IRPSize values is different.";

  assertArrays(values, csr.GetAS(), NZ, "as_ -> Incorrect value at: ");
  assertArrays(correctIRP, &csr.GetIRP()[0], 4, "irp_ -> Incorrect value at: ");
  assertArrays(correctJA, &csr.GetJA()[0], NZ, "ja_ -> Incorrect value at: ");
}

TEST_F(CSRTransformerTest, iostreamTest) {
  const INDEXING_TYPE M = 4, N = 4, NZ = 7;
  std::vector<INDEXING_TYPE> IRP = { 0, 2, 4, 5, 7 }, JA = { 0, 1, 1, 2, 2, 2, 3 };
  std::vector<FLOATING_TYPE> AS = { 11, 12, 22, 23, 33, 43, 44 };
  sspp::representations::CSR expectedCSR(NZ, M, N, IRP, JA, AS);

  std::stringstream stringStream;
  stringStream << expectedCSR;
  sspp::representations::CSR actualCSR;
  stringStream >> actualCSR;

  ASSERT_EQ(expectedCSR.GetRows(), actualCSR.GetRows());
  ASSERT_EQ(expectedCSR.GetColumns(), actualCSR.GetColumns());
  ASSERT_EQ(expectedCSR.GetNonZeros(), actualCSR.GetNonZeros());

  assertArrays(&expectedCSR.GetIRP()[0], &actualCSR.GetIRP()[0], expectedCSR.GetIRP().size(), "Incorrect irp_ at: ");
  assertArrays(&expectedCSR.GetJA()[0], &actualCSR.GetJA()[0], expectedCSR.GetJA().size(), "Incorrect ja_ at: ");
  assertArrays(&expectedCSR.GetAS()[0], &actualCSR.GetAS()[0], expectedCSR.GetAS().size(), "Incorrect as_ at: ");
}