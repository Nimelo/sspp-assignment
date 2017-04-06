#include "PerformanceSpeedupTest.h"
#include <gtest/gtest.h>
#include "TestMacros.h"
#include "TestingUtilities.h"
#include "TestMacros.h"
//#define TEST_MACRO(FIXTURE, NAME) \
//class FIXTURE : public ::testing::Test{};\
// TEST_F(FIXTURE, NAME) {\
//\
//}\



////#define A
////TEST_MACRO(A,X)
//TEST_MACRO(B, Y)
//TEST_MACRO(Z, ere)

TEST_F(PerformanceSpeedupTest, ALWAYS_TRUE) {
  TEST_COUT << "Hello world!\n";
}

TEST_F(ABCDE, ALWAYS_TRUE) {
  TEST_COUT << "Hello world!\n";
  TEST_COUT << this->GetCRS<float>();
  //TEST_COUT << ABCDE::GetKey() << std::endl;
  //  TEST_COUT << ABCDE::ellpack_;
}

TEST_F(ABCDE, ASD) {
  TEST_COUT << "Hello world!\n";
  TEST_COUT << this->GetELLPACK<float>();
  //TEST_COUT << ABCDE::GetKey() << std::endl;
  //  TEST_COUT << ABCDE::ellpack_;
}

//TEST_F(HAHA, asd) {
//  HAHA::stopwatch_.Start();
//}
