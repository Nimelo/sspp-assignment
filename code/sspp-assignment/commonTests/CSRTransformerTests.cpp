#include "CSRTransformerTest.h"
#include <gmock/gmock.h>
#include <gtest\gtest.h>
#include <sstream>
#include <vector>
#include "CRS.h"
#include "MatrixMarket.h"
#include "CRSTransformer.h"

using namespace sspp::common;

TEST_F(CSRTransformerTest, shouldTransformCorrectly_Salvatore) {
  MatrixMarketHeader mmh(Matrix, Sparse, Real, General);
  const unsigned long long M = 4, N = 4, NZ = 7;
  std::vector<unsigned long long> iIndexes = { 0, 0, 1, 1, 2, 3, 3 },
    jIndexes = { 0, 1, 1, 2, 2, 2, 3 };
  std::vector<double> values = { 11, 12, 22, 23, 33, 43, 44 };
  MatrixMarket<double> mm(M, N, NZ, iIndexes, jIndexes, values);

  CRS<float> crs = CRSTransformer::transform<float, double>(mm);

  ASSERT_EQ(M, crs.GetRows());
  ASSERT_EQ(N, crs.GetColumns());
  ASSERT_EQ(NZ, crs.GetNonZeros());
  ASSERT_EQ(NZ, crs.GetColumnIndices().size());
  ASSERT_EQ(NZ, crs.GetValues().size());
  ASSERT_EQ(M + 1, crs.GetRowStartIndexes().size());
  ASSERT_THAT(crs.GetValues(), ::testing::ElementsAre(11, 12, 22, 23, 33, 43, 44));
  ASSERT_THAT(crs.GetColumnIndices(), ::testing::ElementsAre(0, 1, 1, 2, 2, 2, 3));
  ASSERT_THAT(crs.GetRowStartIndexes(), ::testing::ElementsAre(0, 2, 4, 5, 7));
}

TEST_F(CSRTransformerTest, shouldTransformCorrectly) {
  MatrixMarketHeader mmh(Matrix, Sparse, Real, General);
  const unsigned long long M = 3, N = 4, NZ = 5;
  std::vector<unsigned long long> iIndexes = { 0, 1, 1, 2, 2 },
    jIndexes = { 2, 2, 3, 0, 1 };
  std::vector<float> values = { 1, 2, 3, 4, 1 };
  MatrixMarket<float> mm(M, N, NZ, iIndexes, jIndexes, values);

  CRS<float> crs = CRSTransformer::transform(mm);

  ASSERT_EQ(M, crs.GetRows());
  ASSERT_EQ(N, crs.GetColumns());
  ASSERT_EQ(NZ, crs.GetNonZeros());
  ASSERT_EQ(NZ, crs.GetColumnIndices().size());
  ASSERT_EQ(NZ, crs.GetValues().size());
  ASSERT_EQ(M + 1, crs.GetRowStartIndexes().size());
  ASSERT_THAT(crs.GetValues(), ::testing::ElementsAre(1, 2, 3, 4, 1));
  ASSERT_THAT(crs.GetColumnIndices(), ::testing::ElementsAre(2, 2, 3, 0, 1));
  ASSERT_THAT(crs.GetRowStartIndexes(), ::testing::ElementsAre(0, 1, 3, 5));
}

TEST_F(CSRTransformerTest, REAL_GENERAL) {
  MatrixMarketHeader mmh(Matrix, Sparse, Real, General);
  const unsigned long long M = 4, N = 4, NZ = 7;
  std::vector<unsigned long long> iIndexes = { 0, 0, 1, 1, 2, 3, 3 },
    jIndexes = { 0, 1, 1, 2, 2, 2, 3 };
  std::vector<float> values = { 11, 12, 22, 23, 33, 43, 44 };
  MatrixMarket<float> mm(M, N, NZ, iIndexes, jIndexes, values);

  CRS<float> crs = CRSTransformer::transform(mm);

  ASSERT_EQ(M, crs.GetRows());
  ASSERT_EQ(N, crs.GetColumns());
  ASSERT_EQ(NZ, crs.GetNonZeros());
  ASSERT_EQ(NZ, crs.GetColumnIndices().size());
  ASSERT_EQ(NZ, crs.GetValues().size());
  ASSERT_EQ(M + 1, crs.GetRowStartIndexes().size());
  ASSERT_THAT(crs.GetValues(), ::testing::ElementsAre(11, 12, 22, 23, 33, 43, 44));
  ASSERT_THAT(crs.GetColumnIndices(), ::testing::ElementsAre(0, 1, 1, 2, 2, 2, 3));
  ASSERT_THAT(crs.GetRowStartIndexes(), ::testing::ElementsAre(0, 2, 4, 5, 7));
}

TEST_F(CSRTransformerTest, PATTERN_GENEREAL) {
  MatrixMarketHeader mmh(Matrix, Sparse, Pattern, General);
  const unsigned long long M = 4, N = 4, NZ = 7;
  std::vector<unsigned long long> iIndexes = { 0, 0, 1, 1, 2, 3, 3 },
    jIndexes = { 0, 1, 1, 2, 2, 2, 3 };
  MatrixMarket<float> mm(M, N, NZ, iIndexes, jIndexes, std::vector<float>(NZ));

  CRS<float> crs = CRSTransformer::transform(mm);

  ASSERT_EQ(M, crs.GetRows());
  ASSERT_EQ(N, crs.GetColumns());
  ASSERT_EQ(NZ, crs.GetNonZeros());
  ASSERT_EQ(NZ, crs.GetColumnIndices().size());
  ASSERT_EQ(NZ, crs.GetValues().size());
  ASSERT_EQ(M + 1, crs.GetRowStartIndexes().size());
  ASSERT_THAT(crs.GetColumnIndices(), ::testing::ElementsAre(0, 1, 1, 2, 2, 2, 3));
  ASSERT_THAT(crs.GetRowStartIndexes(), ::testing::ElementsAre(0, 2, 4, 5, 7));
}

TEST_F(CSRTransformerTest, REAL_SYMMETRIC) {
  MatrixMarketHeader mmh(Matrix, Sparse, Real, Symetric);
  const unsigned long long M = 4, N = 4, NZ = 7, real_NZ = 10;
  std::vector<unsigned long long> iIndexes = { 0, 0, 1, 1, 2, 3, 3, 1, 2, 2},
    jIndexes = { 0, 1, 1, 2, 2, 2, 3, 0, 1, 3};
  std::vector<float> values = { 11, 12, 22, 23, 33, 43, 44, 12, 23, 43};
  MatrixMarket<float> mm(M, N, real_NZ, iIndexes, jIndexes, values);

  CRS<float> crs = CRSTransformer::transform(mm);

  ASSERT_EQ(M, crs.GetRows());
  ASSERT_EQ(N, crs.GetColumns());
  ASSERT_EQ(real_NZ, crs.GetNonZeros());
  ASSERT_EQ(M + 1, crs.GetRowStartIndexes().size());
  ASSERT_THAT(crs.GetValues(), ::testing::ElementsAre(11, 12, 12, 22, 23, 23, 33, 43, 43, 44));
  ASSERT_THAT(crs.GetColumnIndices(), ::testing::ElementsAre(0, 1, 0, 1, 2, 1, 2, 3, 2, 3));
  ASSERT_THAT(crs.GetRowStartIndexes(), ::testing::ElementsAre(0, 2, 5, 8, 10));
}

TEST_F(CSRTransformerTest, PATTERN_SYMMETRIC) {
  MatrixMarketHeader mmh(Matrix, Sparse, Pattern, Symetric);
  const unsigned long long M = 4, N = 4, NZ = 7, real_NZ = 10;
  std::vector<unsigned long long> iIndexes = { 0, 0, 1, 1, 2, 3, 3, 1, 2, 2 },
    jIndexes = { 0, 1, 1, 2, 2, 2, 3, 0, 1, 3 };
  MatrixMarket<float> mm(M, N, real_NZ, iIndexes, jIndexes, std::vector<float>(real_NZ));

  CRS<float> crs = CRSTransformer::transform(mm);

  ASSERT_EQ(M, crs.GetRows());
  ASSERT_EQ(N, crs.GetColumns());
  ASSERT_EQ(real_NZ, crs.GetNonZeros());
  ASSERT_EQ(M + 1, crs.GetRowStartIndexes().size());
  ASSERT_THAT(crs.GetColumnIndices(), ::testing::ElementsAre(0, 1, 0, 1, 2, 1, 2, 3, 2, 3));
  ASSERT_THAT(crs.GetRowStartIndexes(), ::testing::ElementsAre(0, 2, 5, 8, 10));
}

TEST_F(CSRTransformerTest, iostreamTest) {
  const unsigned long long M = 4, N = 4, NZ = 7;
  std::vector<unsigned long long> IRP = { 0, 2, 4, 5, 7 }, JA = { 0, 1, 1, 2, 2, 2, 3 };
  std::vector<float> AS = { 11, 12, 22, 23, 33, 43, 44 };
  CRS<float> expectedCSR(M, N, NZ, IRP, JA, AS);

  std::stringstream stringStream;
  stringStream << expectedCSR;
  CRS<float> actualCSR;
  stringStream >> actualCSR;

  ASSERT_EQ(expectedCSR.GetRows(), actualCSR.GetRows());
  ASSERT_EQ(expectedCSR.GetColumns(), actualCSR.GetColumns());
  ASSERT_EQ(expectedCSR.GetNonZeros(), actualCSR.GetNonZeros());

  ASSERT_THAT(actualCSR.GetValues(), ::testing::ElementsAre(11, 12, 22, 23, 33, 43, 44));
  ASSERT_THAT(actualCSR.GetColumnIndices(), ::testing::ElementsAre(0, 1, 1, 2, 2, 2, 3));
  ASSERT_THAT(actualCSR.GetRowStartIndexes(), ::testing::ElementsAre(0, 2, 4, 5, 7));
}