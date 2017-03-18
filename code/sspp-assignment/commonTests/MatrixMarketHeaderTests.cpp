#include "MatrixMarketHeaderTest.h"
#include "../common/MatrixMarketHeader.h"
#include <sstream>
using namespace sspp::io;

TEST_F(MatrixMarketHeaderTest, shoudLoadCorrectly) {
  std::stringstream ss;
  ss << MatrixMarketHeader::MatrixMarketBanner_STR << " "
    << MatrixMarketHeader::MM_MTX_STR << " "
    << MatrixMarketHeader::MM_DENSE_STR << " "
    << MatrixMarketHeader::MM_REAL_STR << " "
    << MatrixMarketHeader::MM_GENERAL_STR;

  MatrixMarketHeader mmh;
  auto result = mmh.Load(ss);

  ASSERT_EQ(MatrixMarketErrorCodes::SUCCESS, result);
  ASSERT_TRUE(mmh.IsMatrix());
  ASSERT_TRUE(mmh.IsDense());
  ASSERT_TRUE(mmh.IsReal());
  ASSERT_TRUE(mmh.IsGeneral());
  ASSERT_TRUE(mmh.IsValid());
}