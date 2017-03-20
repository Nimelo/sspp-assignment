#include "CSRTransformerTest.h"
#include "Definitions.h"
#include <gtest\gtest.h>
#include <sstream>
#include <vector>
#include <string>

TEST_F(CSRTransformerTest, shouldTransformCorrectly_Salvatore) {
  const int M = 4, N = 4, NZ = 7;
  auto iIndexes = new std::vector<INDEXING_TYPE>{ 0, 0, 1, 1, 2, 3, 3 },
    jIndexes = new std::vector<INDEXING_TYPE>{ 0, 1, 1, 2, 2, 2, 3 };
  auto values = new std::vector<FLOATING_TYPE>{ 11, 12, 22, 23, 33, 43, 44 };

  std::vector<INDEXING_TYPE> correctIRP = { 0, 2, 4, 5, 7 },
    correctJA = { 0, 1, 1, 2, 2, 2, 3 };

  sspp::representations::IntermediarySparseMatrix ism(M, N, NZ, iIndexes, jIndexes, values);
  auto csr = csrTransformer->Transform(ism);

  ASSERT_EQ(M, csr->GetRows()) << "rows_ values is different.";
  ASSERT_EQ(N, csr->GetColumns()) << "columns_ values is different.";
  ASSERT_EQ(NZ, csr->GetJA()->size()) << "JASize values is different.";
  ASSERT_EQ(NZ, csr->GetAS()->size()) << "ASSize values is different.";
  ASSERT_EQ(M + 1, csr->GetIRP()->size()) << "IRPSize values is different.";

  assertArrays(*values, *csr->GetAS(), NZ, "as_ -> Incorrect value at: ");
  assertArrays(correctIRP, *csr->GetIRP(), 5, "irp_ -> Incorrect value at: ");
  assertArrays(correctJA, *csr->GetJA(), NZ, "ja_ -> Incorrect value at: ");
  delete csr;
}

TEST_F(CSRTransformerTest, shouldTransformCorrectly) {
  const int M = 3, N = 4, NZ = 5;
  auto iIndexes = new std::vector<INDEXING_TYPE>{ 0, 1, 1, 2, 2 },
    jIndexes = new std::vector<INDEXING_TYPE>{ 2, 2, 3, 0, 1 };
  auto values = new std::vector<FLOATING_TYPE>{ 1, 2, 3, 4, 1 };
  std::vector<INDEXING_TYPE> correctIRP = { 0, 1, 3, 5 },
    correctJA = { 2, 2, 3, 0, 1 };

  sspp::representations::IntermediarySparseMatrix ism(M, N, NZ, iIndexes, jIndexes, values);
  auto csr = csrTransformer->Transform(ism);

  ASSERT_EQ(M, csr->GetRows()) << "rows_ values is different.";
  ASSERT_EQ(N, csr->GetColumns()) << "columns_ values is different.";
  ASSERT_EQ(NZ, csr->GetJA()->size()) << "JASize values is different.";
  ASSERT_EQ(NZ, csr->GetAS()->size()) << "ASSize values is different.";
  ASSERT_EQ(M + 1, csr->GetIRP()->size()) << "IRPSize values is different.";

  assertArrays(*values, *csr->GetAS(), NZ, "as_ -> Incorrect value at: ");
  assertArrays(correctIRP, *csr->GetIRP(), 4, "irp_ -> Incorrect value at: ");
  assertArrays(correctJA, *csr->GetJA(), NZ, "ja_ -> Incorrect value at: ");
  delete csr;
}

TEST_F(CSRTransformerTest, iostreamTest) {
  const INDEXING_TYPE M = 4, N = 4, NZ = 7;
  auto IRP = new std::vector<INDEXING_TYPE>{ 0, 2, 4, 5, 7 }, JA = new std::vector<INDEXING_TYPE>{ 0, 1, 1, 2, 2, 2, 3 };
  auto AS = new std::vector<FLOATING_TYPE>{ 11, 12, 22, 23, 33, 43, 44 };
  sspp::representations::CSR expectedCSR(NZ, M, N, IRP, JA, AS);

  std::stringstream stringStream;
  stringStream << expectedCSR;
  sspp::representations::CSR actualCSR;
  stringStream >> actualCSR;

  ASSERT_EQ(expectedCSR.GetRows(), actualCSR.GetRows());
  ASSERT_EQ(expectedCSR.GetColumns(), actualCSR.GetColumns());
  ASSERT_EQ(expectedCSR.GetNonZeros(), actualCSR.GetNonZeros());

  assertArrays(expectedCSR.GetIRP(), actualCSR.GetIRP(), expectedCSR.GetIRP()->size(), "Incorrect irp_ at: ");
  assertArrays(expectedCSR.GetJA(), actualCSR.GetJA(), expectedCSR.GetJA()->size(), "Incorrect ja_ at: ");
  assertArrays(expectedCSR.GetAS(), actualCSR.GetAS(), expectedCSR.GetAS()->size(), "Incorrect as_ at: ");
}