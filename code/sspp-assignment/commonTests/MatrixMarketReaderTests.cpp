#include "MatrixMarketReaderTest.h"
#include <fstream>
#include "..\common\mmio.h"
#include "..\common\Definitions.h"

TEST_F(MatrixMarketReaderTest, CRG) {
  const char * fileName = "matrixMarketTestCRG";
  const INDEXING_TYPE M = 5, N = 5, NZ = 2;
  FLOATING_TYPE VAL[] = { 0, 1 };
  INDEXING_TYPE I[] = { 5, 4 }, J[] = { 3, 2 };

  std::fstream fs;
  fs.open(fileName, std::fstream::out | std::fstream::trunc);
  fs << MatrixMarketBanner << SPACE
    << MM_MTX_STR << SPACE
    << MM_COORDINATE_STR << SPACE
    << MM_REAL_STR << SPACE
    << MM_GENERAL_STR << std::endl;
  fs << M << SPACE << N << SPACE << NZ << std::endl;
  for(int i = 0; i < NZ; i++) {
    fs << I[i] << SPACE << J[i] << SPACE << VAL[i] << std::endl;
  }
  fs.close();

  try {
    auto ism = this->matrixMarketReader->FromFile(fileName);

    ASSERT_EQ(M, ism.GetRows());
    ASSERT_EQ(N, ism.GetColumns());
    ASSERT_EQ(NZ, ism.GetNonZeros());
    assertArrays(I, &ism.GetRowIndexes()[0], NZ, "");
    assertArrays(J, &ism.GetColumnIndexes()[0], NZ, "");
    assertArrays(VAL, &ism.GetValues()[0], NZ, "");
  } catch(...) {
    std::remove(fileName);
    FAIL();
  }
  std::remove(fileName);
}

TEST_F(MatrixMarketReaderTest, CRS) {
  const char * fileName = "matrixMarketTestCRS";
  const INDEXING_TYPE M = 5, N = 5, NZ = 2;
  FLOATING_TYPE VAL[] = { 0, 1 };
  INDEXING_TYPE I[] = { 5, 4 };
  INDEXING_TYPE J[] = { 3, 2 };

  std::fstream fs;
  fs.open(fileName, std::fstream::out | std::fstream::trunc);
  fs << MatrixMarketBanner << SPACE
    << MM_MTX_STR << SPACE
    << MM_COORDINATE_STR << SPACE
    << MM_REAL_STR << SPACE
    << MM_SYMM_STR << std::endl;
  fs << M << SPACE << N << SPACE << NZ << std::endl;
  for(int i = 0; i < NZ; i++) {
    fs << I[i] << SPACE << J[i] << SPACE << VAL[i] << std::endl;
  }
  fs.close();

  try {
    auto ism = this->matrixMarketReader->FromFile(fileName);

    ASSERT_EQ(M, ism.GetRows());
    ASSERT_EQ(N, ism.GetColumns());
    ASSERT_EQ(2*NZ, ism.GetNonZeros());
    assertArrays(I, &ism.GetRowIndexes()[0], NZ, "");
    assertArrays(J, &ism.GetColumnIndexes()[0], NZ, "");
    assertArrays(VAL, &ism.GetValues()[0], NZ, "");

    assertArrays(I, &ism.GetColumnIndexes()[0] + NZ, NZ, "");
    assertArrays(J, &ism.GetRowIndexes()[0] + NZ, NZ, "");
    assertArrays(VAL, &ism.GetValues()[0] + NZ, NZ, "");
  } catch(...) {
    std::remove(fileName);
    FAIL();
  }
  std::remove(fileName);
}


TEST_F(MatrixMarketReaderTest, CPS) {
  const char * fileName = "matrixMarketTestCPS";
  const INDEXING_TYPE M = 5, N = 5, NZ = 2;
  INDEXING_TYPE I[] = { 5, 4 }, J[] = { 3, 2 };

  std::fstream fs;
  fs.open(fileName, std::fstream::out | std::fstream::trunc);
  fs << MatrixMarketBanner << SPACE
    << MM_MTX_STR << SPACE
    << MM_COORDINATE_STR << SPACE
    << MM_PATTERN_STR << SPACE
    << MM_SYMM_STR << std::endl;
  fs << M << SPACE << N << SPACE << NZ << std::endl;
  for(int i = 0; i < NZ; i++) {
    fs << I[i] << SPACE << J[i] << SPACE << std::endl;
  }
  fs.close();

  try {
    auto ism = this->matrixMarketReader->FromFile(fileName);

    ASSERT_EQ(M, ism.GetRows());
    ASSERT_EQ(N, ism.GetColumns());
    ASSERT_EQ(2*NZ, ism.GetNonZeros());
    assertArrays(I, &ism.GetRowIndexes()[0], NZ, "");
    assertArrays(J, &ism.GetColumnIndexes()[0], NZ, "");

    assertArrays(I, &ism.GetColumnIndexes()[0] + NZ, NZ, "");
    assertArrays(J, &ism.GetRowIndexes()[0] + NZ, NZ, "");
    assertArrays(&ism.GetValues()[0], &ism.GetValues()[0] + NZ, NZ, "");
  } catch(...) {
    std::remove(fileName);
    FAIL();
  }
  std::remove(fileName);
}

TEST_F(MatrixMarketReaderTest, CPG) {
  const char * fileName = "matrixMarketTestCPG";
  const INDEXING_TYPE M = 5, N = 5, NZ = 2;
  INDEXING_TYPE I[] = { 5, 4 }, J[] = { 3, 2 };

  std::fstream fs;
  fs.open(fileName, std::fstream::out | std::fstream::trunc);
  fs << MatrixMarketBanner << SPACE
    << MM_MTX_STR << SPACE
    << MM_COORDINATE_STR << SPACE
    << MM_PATTERN_STR << SPACE
    << MM_GENERAL_STR << std::endl;
  fs << M << SPACE << N << SPACE << NZ << std::endl;
  for(int i = 0; i < NZ; i++) {
    fs << I[i] << SPACE << J[i] << SPACE << std::endl;
  }
  fs.close();

  try {
    auto ism = this->matrixMarketReader->FromFile(fileName);

    ASSERT_EQ(M, ism.GetRows());
    ASSERT_EQ(N, ism.GetColumns());
    ASSERT_EQ(NZ, ism.GetNonZeros());
    assertArrays(I, &ism.GetRowIndexes()[0], NZ, "");
    assertArrays(J, &ism.GetColumnIndexes()[0], NZ, "");
  } catch(...) {
    std::remove(fileName);
    FAIL();
  }
  std::remove(fileName);
}