#include "ELLPACKTransformerTest.h"
#include "Definitions.h"
#include <gtest\gtest.h>
#include <sstream>

TEST_F(ELLPACKTransformerTest, shouldSolveCorrectly_Salvatore) {
  const INDEXING_TYPE M = 4, N = 4, NZ = 7, correctMAXNZ = 2;
  std::vector<INDEXING_TYPE> iIndexes = { 0, 0, 1, 1, 2, 3, 3 },
    jIndexes = { 0, 1, 1, 2, 2, 2, 3 };
  std::vector<FLOATING_TYPE> values = { 11, 12, 22, 23, 33, 43, 44 };
  std::vector<FLOATING_TYPE> correctAS = { 11, 12, 22, 23, 33, 0, 43, 44 };
  std::vector<INDEXING_TYPE> correctJA = { 0, 1, 1, 2, 2, 2, 2, 3 };

  sspp::representations::IntermediarySparseMatrix ism(M, N, NZ, iIndexes, jIndexes, values);
  auto ellpack = ellpackTransformer->Transform(ism);

  ASSERT_EQ(M, ellpack.GetRows()) << "rows_ values is different.";
  ASSERT_EQ(N, ellpack.GetColumns()) << "columns_ values is different.";
  ASSERT_EQ(NZ, ellpack.GetNonZeros()) << "non_zeros_ values is different.";
  ASSERT_EQ(correctMAXNZ, ellpack.GetMaxRowNonZeros()) << "max_row_non_zeros_ values is different.";
  assertArrays(&correctAS[0], &ellpack.GetAS()[0], correctAS.size(), "as_ values is different.");
  assertArrays(&correctJA[0], &ellpack.GetJA()[0], correctJA.size(), "ja_ values is different.");
}

TEST_F(ELLPACKTransformerTest, shouldSolveCorrectly) {
  const INDEXING_TYPE M = 3, N = 4, NZ = 5, correctMAXNZ = 3;
  std::vector<INDEXING_TYPE> iIndexes = { 0, 0, 0, 1, 2 },
    jIndexes = { 0, 1, 2, 2, 3 };
  std::vector<FLOATING_TYPE> values = { 2, 7, 1, 4, 1 };
  std::vector<FLOATING_TYPE> correctAS = { 2, 7, 1, 4, 0, 0, 1, 0, 0 };
  std::vector<INDEXING_TYPE> correctJA = { 0, 1, 2, 2, 2, 2, 3, 3, 3};

  sspp::representations::IntermediarySparseMatrix ism(M, N, NZ, iIndexes, jIndexes, values);
  auto ellpack = ellpackTransformer->Transform(ism);

  ASSERT_EQ(M, ellpack.GetRows()) << "rows_ values is different.";
  ASSERT_EQ(N, ellpack.GetColumns()) << "columns_ values is different.";
  ASSERT_EQ(NZ, ellpack.GetNonZeros()) << "non_zeros_ values is different.";
  ASSERT_EQ(correctMAXNZ, ellpack.GetMaxRowNonZeros()) << "max_row_non_zeros_ values is different.";
  assertArrays(&correctAS[0], &ellpack.GetAS()[0], correctAS.size(), "as_ values is different.");
  assertArrays(&correctJA[0], &ellpack.GetJA()[0], correctJA.size(), "ja_ values is different.");
}

TEST_F(ELLPACKTransformerTest, iostreamTest) {
  const INDEXING_TYPE M = 4, N = 4, NZ = 7, MAXNZ = 2;
  std::vector<FLOATING_TYPE> AS = { 11, 12, 22, 23, 33, 0, 43, 44 };
  std::vector<INDEXING_TYPE> JA = { 0, 1, 1, 2, 2, 2, 2, 3 };
  sspp::representations::ELLPACK ellpack(M, N, NZ, MAXNZ, JA, AS);

  std::stringstream stringStream;
  stringStream << ellpack;
  sspp::representations::ELLPACK actualEllpack;
  stringStream >> actualEllpack;

  ASSERT_EQ(ellpack.GetRows(), actualEllpack.GetRows()) << "rows_ values is different.";
  ASSERT_EQ(ellpack.GetColumns(), actualEllpack.GetColumns()) << "columns_ values is different.";
  ASSERT_EQ(ellpack.GetNonZeros(), actualEllpack.GetNonZeros()) << "non_zeros_ values is different.";
  ASSERT_EQ(ellpack.GetMaxRowNonZeros(), actualEllpack.GetMaxRowNonZeros()) << "max_row_non_zeros_ values is different.";
  assertArrays(&ellpack.GetAS()[0], &actualEllpack.GetAS()[0], ellpack.GetAS().size(), "as_ values is different.");
  assertArrays(&ellpack.GetJA()[0], &actualEllpack.GetJA()[0], ellpack.GetJA().size(), "ja_ values is different.");
}