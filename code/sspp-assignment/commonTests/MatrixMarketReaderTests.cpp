#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "MatrixMarketReaderTest.h"
#include "MatrixMarketHeader.h"
#include "MatrixMarket.h"
#include "MarketMatrixReader.h"
#include "CRSTransformer.h"
#include "FloatPatternResolver.h"

using namespace sspp::common;

TEST_F(MatrixMarketReaderTest, REAL_SYMMETRIC) {
  MatrixMarketHeader mmh(Matrix, Sparse, Real, Symetric);
  const unsigned M = 4, N = 4, NZ = 7, real_NZ = 10;
  std::vector<unsigned> iIndexes = { 1, 1, 2, 2, 3, 4, 4 },
    jIndexes = { 1, 2, 2, 3, 3, 3, 4 };
  std::vector<float> values = { 11, 12, 22, 23, 33, 43, 44 };
  WriteHeader(mmh);
  WriteIndices(M, N, NZ, iIndexes, jIndexes, values);

  FloatPatternResolver resolver;
  auto mm = MatrixMarketReader::Read(ss, resolver);

  ASSERT_EQ(M, mm.GetRows());
  ASSERT_EQ(N, mm.GetColumns());
  ASSERT_EQ(real_NZ, mm.GetNonZeros());
  ASSERT_THAT(mm.GetValues(), ::testing::UnorderedElementsAre(11, 12, 12, 22, 23, 23, 33, 43, 43, 44));
  ASSERT_THAT(mm.GetColumnIndices(), ::testing::UnorderedElementsAre(0, 1, 0, 1, 2, 1, 2, 3, 2, 3));
  ASSERT_THAT(mm.GetRowIndices(), ::testing::UnorderedElementsAre(0, 0, 1, 1, 1, 2, 2, 2, 3, 3));
}

TEST_F(MatrixMarketReaderTest, PATTERN_SYMMETRIC) {
  sspp::common::MatrixMarketHeader mmh(Matrix, Sparse, Pattern, Symetric);
  const unsigned M = 4, N = 4, NZ = 7, real_NZ = 10;
  std::vector<unsigned> iIndexes = { 1, 1, 2, 2, 3, 4, 4 },
    jIndexes = { 1, 2, 2, 3, 3, 3, 4 };
  WriteHeader(mmh);
  WriteIndicesPattern(M, N, NZ, iIndexes, jIndexes);

  FloatPatternResolver resolver;
  auto mm = MatrixMarketReader::Read(ss, resolver);

  ASSERT_EQ(M, mm.GetRows());
  ASSERT_EQ(N, mm.GetColumns());
  ASSERT_EQ(real_NZ, mm.GetNonZeros());
  ASSERT_THAT(mm.GetColumnIndices(), ::testing::UnorderedElementsAre(0, 1, 0, 1, 2, 1, 2, 3, 2, 3));
  ASSERT_THAT(mm.GetRowIndices(), ::testing::UnorderedElementsAre(0, 0, 1, 1, 1, 2, 2, 2, 3, 3));
}