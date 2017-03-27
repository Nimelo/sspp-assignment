#include "CSRTransformerTest.h"
#include <gmock/gmock.h>
#include <gtest\gtest.h>
#include <sstream>
#include <vector>
#include "UnsignedFloatReader.h"
#include "CRS.h"

using namespace sspp::common;

TEST_F(CSRTransformerTest, shouldTransformCorrectly_Salvatore) {
  MatrixMarketHeader mmh(Matrix, Sparse, Real, General);
  const unsigned M = 4, N = 4, NZ = 7;
  std::vector<unsigned> iIndexes = { 1, 1, 2, 2, 3, 4, 4 },
    jIndexes = { 1, 2, 2, 3, 3, 3, 4 };
  std::vector<float> values = { 11, 12, 22, 23, 33, 43, 44 };
  WriteHeader(mmh);
  WriteIndices<unsigned, unsigned, float>(M, N, NZ, iIndexes, jIndexes, values);
  UnsignedFlaotReader reader;
  MatrixMarketStream<float> mms(ss, reader);

  CRS<float> crs(mms);

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
  const unsigned M = 3, N = 4, NZ = 5;
  std::vector<unsigned> iIndexes = { 1, 2, 2, 3, 3 },
    jIndexes = { 3, 3, 4, 1, 2 };
  std::vector<float> values = { 1, 2, 3, 4, 1 };
  WriteHeader(mmh);
  WriteIndices<unsigned, unsigned, float>(M, N, NZ, iIndexes, jIndexes, values);
  UnsignedFlaotReader reader;
  MatrixMarketStream<float> mms(ss, reader);

  CRS<float> crs(mms);

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
  const unsigned M = 4, N = 4, NZ = 7;
  std::vector<unsigned> iIndexes = { 1, 1, 2, 2, 3, 4, 4 },
    jIndexes = { 1, 2, 2, 3, 3, 3, 4 };
  std::vector<float> values = { 11, 12, 22, 23, 33, 43, 44 };
  WriteHeader(mmh);
  WriteIndices<unsigned, unsigned, float>(M, N, NZ, iIndexes, jIndexes, values);
  UnsignedFlaotReader reader;
  MatrixMarketStream<float> mms(ss, reader);

  CRS<float> crs(mms);

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
  const unsigned M = 4, N = 4, NZ = 7;
  std::vector<unsigned> iIndexes = { 1, 1, 2, 2, 3, 4, 4 },
    jIndexes = { 1, 2, 2, 3, 3, 3, 4 };
  WriteHeader(mmh);
  WriteIndicesPattern<unsigned, unsigned>(M, N, NZ, iIndexes, jIndexes);
  UnsignedFlaotReader reader;
  MatrixMarketStream<float> mms(ss, reader);

  CRS<float> crs(mms);

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
  const unsigned M = 4, N = 4, NZ = 7, real_NZ = 10;
  std::vector<unsigned> iIndexes = { 1, 1, 2, 2, 3, 4, 4 },
    jIndexes = { 1, 2, 2, 3, 3, 3, 4 };
  std::vector<float> values = { 11, 12, 22, 23, 33, 43, 44 };
  WriteHeader(mmh);
  WriteIndices<unsigned, unsigned, float>(M, N, NZ, iIndexes, jIndexes, values);
  UnsignedFlaotReader reader;
  MatrixMarketStream<float> mms(ss, reader);

  CRS<float> crs(mms);

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
  const unsigned M = 4, N = 4, NZ = 7, real_NZ = 10;
  std::vector<unsigned> iIndexes = { 1, 1, 2, 2, 3, 4, 4 },
    jIndexes = { 1, 2, 2, 3, 3, 3, 4 };
  WriteHeader(mmh);
  WriteIndicesPattern<unsigned, unsigned>(M, N, NZ, iIndexes, jIndexes);
  UnsignedFlaotReader reader;
  MatrixMarketStream<float> mms(ss, reader);

  CRS<float> crs(mms);

  ASSERT_EQ(M, crs.GetRows());
  ASSERT_EQ(N, crs.GetColumns());
  ASSERT_EQ(real_NZ, crs.GetNonZeros());
  ASSERT_EQ(M + 1, crs.GetRowStartIndexes().size());
  ASSERT_THAT(crs.GetColumnIndices(), ::testing::ElementsAre(0, 1, 0, 1, 2, 1, 2, 3, 2, 3));
  ASSERT_THAT(crs.GetRowStartIndexes(), ::testing::ElementsAre(0, 2, 5, 8, 10));
}

TEST_F(CSRTransformerTest, iostreamTest) {
  const unsigned M = 4, N = 4, NZ = 7;
  std::vector<unsigned> IRP = { 0, 2, 4, 5, 7 }, JA = { 0, 1, 1, 2, 2, 2, 3 };
  std::vector<float> AS = { 11, 12, 22, 23, 33, 43, 44 };
  CRS<float> expectedCSR(NZ, M, N, IRP, JA, AS);
  
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