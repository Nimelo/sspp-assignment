#include "ELLPACKTransformerTest.h"
#include <gtest\gtest.h>
#include <gmock/gmock.h>
#include <sstream>
#include <gmock/gmock.h>
#include "ELLPACK.h"
#include "MatrixMarketHeader.h"
#include "MatrixMarket.h"
#include "ELLPACKTransformer.h"

using namespace sspp::common;

TEST_F(ELLPACKTransformerTest, TRANSFORM_1) {
  MatrixMarketHeader mmh(Matrix, Sparse, Real, General);
  const unsigned M = 4, N = 4, NZ = 7, correctMAXNZ = 2;
  std::vector<unsigned> iIndexes = { 0, 0, 1, 1, 2, 3, 3 },
    jIndexes = { 0, 1, 1, 2, 2, 2, 3 };
  std::vector<float> values = { 11, 12, 22, 23, 33, 43, 44 };
  std::vector<float> correctAS = { 11, 12, 22, 23, 33, 0, 43, 44 };
  std::vector<unsigned> correctJA = { 0, 1, 1, 2, 2, 2, 2, 3 };
  MatrixMarket<float> mm(M, N, NZ, iIndexes, jIndexes, values);

  ELLPACK<float> ellpack = ELLPACKTransformer::transform(mm);

  ASSERT_EQ(M, ellpack.GetRows());
  ASSERT_EQ(N, ellpack.GetColumns());
  ASSERT_EQ(NZ, ellpack.GetNonZeros());
  ASSERT_EQ(correctMAXNZ, ellpack.GetMaxRowNonZeros());
  ASSERT_THAT(ellpack.GetValues(), ::testing::UnorderedElementsAre(12, 11, 23, 22, 33, 0, 44, 43));
  ASSERT_THAT(ellpack.GetColumnIndices(), ::testing::UnorderedElementsAre(1, 0, 2, 1, 2, 0, 3, 2));
}

TEST_F(ELLPACKTransformerTest, TRANSFORM_2) {
  MatrixMarketHeader mmh(Matrix, Sparse, Real, General);
  const unsigned M = 3, N = 4, NZ = 5, correctMAXNZ = 3;
  std::vector<unsigned> iIndexes = { 0, 0, 0, 1, 2 },
    jIndexes = { 0, 1, 2, 2, 3 };
  std::vector<float> values = { 2, 7, 1, 4, 1 };
  std::vector<float> correctAS = { 2, 7, 1, 4, 0, 0, 1, 0, 0 };
  std::vector<unsigned> correctJA = { 0, 1, 2, 2, 2, 2, 3, 3, 3 };
  MatrixMarket<float> mm(M, N, NZ, iIndexes, jIndexes, values);

  ELLPACK<float> ellpack = ELLPACKTransformer::transform(mm);
  ASSERT_EQ(M, ellpack.GetRows());
  ASSERT_EQ(N, ellpack.GetColumns());
  ASSERT_EQ(NZ, ellpack.GetNonZeros());
  ASSERT_EQ(correctMAXNZ, ellpack.GetMaxRowNonZeros());
  ASSERT_THAT(ellpack.GetValues(), ::testing::UnorderedElementsAre(1, 7, 2, 4, 0, 0, 1, 0, 0));
  ASSERT_THAT(ellpack.GetColumnIndices(), ::testing::UnorderedElementsAre(2, 1, 0, 2, 0, 0, 3, 0, 0));
}

TEST_F(ELLPACKTransformerTest, iostreamTest) {
  const unsigned M = 4, N = 4, NZ = 7, MAXNZ = 2;
  std::vector<float> AS = { 11, 12, 22, 23, 33, 0, 43, 44 };
  std::vector<unsigned> JA = { 0, 1, 1, 2, 2, 2, 2, 3 };
  ELLPACK<float> ellpack(M, N, NZ, MAXNZ, JA, AS);

  std::stringstream stringStream;
  stringStream << ellpack;
  ELLPACK<float> actualEllpack;
  stringStream >> actualEllpack;

  ASSERT_EQ(ellpack.GetRows(), actualEllpack.GetRows()) << "rows_ values is different.";
  ASSERT_EQ(ellpack.GetColumns(), actualEllpack.GetColumns()) << "columns_ values is different.";
  ASSERT_EQ(ellpack.GetNonZeros(), actualEllpack.GetNonZeros()) << "non_zeros_ values is different.";
  ASSERT_EQ(ellpack.GetMaxRowNonZeros(), actualEllpack.GetMaxRowNonZeros()) << "max_row_non_zeros_ values is different.";
  ASSERT_THAT(ellpack.GetValues(), ::testing::ContainerEq(actualEllpack.GetValues()));
  ASSERT_THAT(ellpack.GetColumnIndices(), ::testing::ContainerEq(actualEllpack.GetColumnIndices()));
}