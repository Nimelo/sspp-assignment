#include "CSRTransformerTest.h"
#include <gmock/gmock.h>
#include <gtest\gtest.h>
#include <sstream>
#include <vector>
#include "CRS.h"
#include "MatrixMarket.h"
#include "CRSTransformer.h"

using namespace sspp::common;


TEST_F(CSRTransformerTest, ALL_ROWS) {
  MatrixMarketHeader mmh(Matrix, Sparse, Real, General);
  const unsigned M = 3, N = 3, NZ = 5;
  std::vector<unsigned> iIndexes = { 0, 0, 1, 1, 2 },
    jIndexes = { 0, 1, 1, 2, 2 };
  std::vector<float> values = { 17, 4, 5, 2, 11 };
  MatrixMarket<float> mm(M, N, NZ, iIndexes, jIndexes, values);

  CRS<float> crs = CRSTransformer::transform(mm);

  ASSERT_EQ(M, crs.GetRows());
  ASSERT_EQ(N, crs.GetColumns());
  ASSERT_EQ(NZ, crs.GetNonZeros());
  ASSERT_EQ(NZ, crs.GetColumnIndices().size());
  ASSERT_EQ(NZ, crs.GetValues().size());
  ASSERT_EQ(M + 1, crs.GetRowStartIndexes().size());
  ASSERT_THAT(crs.GetValues(), ::testing::ElementsAre(17, 4, 5, 2, 11));
  ASSERT_THAT(crs.GetColumnIndices(), ::testing::ElementsAre(0, 1, 1, 2, 2));
  ASSERT_THAT(crs.GetRowStartIndexes(), ::testing::ElementsAre(0, 2, 4, 5));
}

TEST_F(CSRTransformerTest, NO_MIDDLE_ROW) {
  MatrixMarketHeader mmh(Matrix, Sparse, Real, General);
  const unsigned M = 3, N = 3, NZ = 3;
  std::vector<unsigned> iIndexes = { 0, 0, 2 },
    jIndexes = { 0, 1, 2 };
  std::vector<float> values = { 17, 4, 11 };
  MatrixMarket<float> mm(M, N, NZ, iIndexes, jIndexes, values);

  CRS<float> crs = CRSTransformer::transform(mm);

  ASSERT_EQ(M, crs.GetRows());
  ASSERT_EQ(N, crs.GetColumns());
  ASSERT_EQ(NZ, crs.GetNonZeros());
  ASSERT_EQ(NZ, crs.GetColumnIndices().size());
  ASSERT_EQ(NZ, crs.GetValues().size());
  ASSERT_EQ(M + 1, crs.GetRowStartIndexes().size());
  ASSERT_THAT(crs.GetValues(), ::testing::ElementsAre(17, 4, 11));
  ASSERT_THAT(crs.GetColumnIndices(), ::testing::ElementsAre(0, 1, 2));
  ASSERT_THAT(crs.GetRowStartIndexes(), ::testing::ElementsAre(0, 2, 2, 3));
}

TEST_F(CSRTransformerTest, NO_MIDDLE_ROW_2) {
  MatrixMarketHeader mmh(Matrix, Sparse, Real, General);
  const unsigned M = 3, N = 3, NZ = 2;
  std::vector<unsigned> iIndexes = { 0, 2 },
    jIndexes = { 1, 2 };
  std::vector<float> values = { 4, 11 };
  MatrixMarket<float> mm(M, N, NZ, iIndexes, jIndexes, values);

  CRS<float> crs = CRSTransformer::transform(mm);

  ASSERT_EQ(M, crs.GetRows());
  ASSERT_EQ(N, crs.GetColumns());
  ASSERT_EQ(NZ, crs.GetNonZeros());
  ASSERT_EQ(NZ, crs.GetColumnIndices().size());
  ASSERT_EQ(NZ, crs.GetValues().size());
  ASSERT_EQ(M + 1, crs.GetRowStartIndexes().size());
  ASSERT_THAT(crs.GetValues(), ::testing::ElementsAre(4, 11));
  ASSERT_THAT(crs.GetColumnIndices(), ::testing::ElementsAre(1, 2));
  ASSERT_THAT(crs.GetRowStartIndexes(), ::testing::ElementsAre(0, 1, 1, 2));
}

TEST_F(CSRTransformerTest, ONE_VALUE_ONLY) {
  MatrixMarketHeader mmh(Matrix, Sparse, Real, General);
  const unsigned M = 3, N = 3, NZ = 1;
  std::vector<unsigned> iIndexes = { 2 },
    jIndexes = { 2 };
  std::vector<float> values = { 11 };
  MatrixMarket<float> mm(M, N, NZ, iIndexes, jIndexes, values);

  CRS<float> crs = CRSTransformer::transform(mm);

  ASSERT_EQ(M, crs.GetRows());
  ASSERT_EQ(N, crs.GetColumns());
  ASSERT_EQ(NZ, crs.GetNonZeros());
  ASSERT_EQ(NZ, crs.GetColumnIndices().size());
  ASSERT_EQ(NZ, crs.GetValues().size());
  ASSERT_EQ(M + 1, crs.GetRowStartIndexes().size());
  ASSERT_THAT(crs.GetValues(), ::testing::ElementsAre(11));
  ASSERT_THAT(crs.GetColumnIndices(), ::testing::ElementsAre(2));
  ASSERT_THAT(crs.GetRowStartIndexes(), ::testing::ElementsAre(0, 1, 1, 1));
}

TEST_F(CSRTransformerTest, FIRST_AND_LAST_ROW) {
  MatrixMarketHeader mmh(Matrix, Sparse, Real, General);
  const unsigned M = 4, N = 4, NZ = 2;
  std::vector<unsigned> iIndexes = { 0, 3 },
    jIndexes = { 0, 3 };
  std::vector<float> values = { 4, 11};
  MatrixMarket<float> mm(M, N, NZ, iIndexes, jIndexes, values);

  CRS<float> crs = CRSTransformer::transform(mm);

  ASSERT_EQ(M, crs.GetRows());
  ASSERT_EQ(N, crs.GetColumns());
  ASSERT_EQ(NZ, crs.GetNonZeros());
  ASSERT_EQ(NZ, crs.GetColumnIndices().size());
  ASSERT_EQ(NZ, crs.GetValues().size());
  ASSERT_EQ(M + 1, crs.GetRowStartIndexes().size());
  ASSERT_THAT(crs.GetValues(), ::testing::ElementsAre(4, 11));
  ASSERT_THAT(crs.GetColumnIndices(), ::testing::ElementsAre(0, 3));
  ASSERT_THAT(crs.GetRowStartIndexes(), ::testing::ElementsAre(0, 1, 1, 1, 2));
}

TEST_F(CSRTransformerTest, FIRST_AND_LAST_ROW_2) {
  MatrixMarketHeader mmh(Matrix, Sparse, Real, General);
  const unsigned M = 4, N = 4, NZ = 4;
  std::vector<unsigned> iIndexes = { 0, 0, 0, 3 },
    jIndexes = { 0, 1, 2, 3 };
  std::vector<float> values = { 4, 5, 6, 11 };
  MatrixMarket<float> mm(M, N, NZ, iIndexes, jIndexes, values);

  CRS<float> crs = CRSTransformer::transform(mm);

  ASSERT_EQ(M, crs.GetRows());
  ASSERT_EQ(N, crs.GetColumns());
  ASSERT_EQ(NZ, crs.GetNonZeros());
  ASSERT_EQ(NZ, crs.GetColumnIndices().size());
  ASSERT_EQ(NZ, crs.GetValues().size());
  ASSERT_EQ(M + 1, crs.GetRowStartIndexes().size());
  ASSERT_THAT(crs.GetValues(), ::testing::ElementsAre(4, 5, 6, 11));
  ASSERT_THAT(crs.GetColumnIndices(), ::testing::ElementsAre(0, 1, 2, 3));
  ASSERT_THAT(crs.GetRowStartIndexes(), ::testing::ElementsAre(0, 3, 3, 3, 4));
}