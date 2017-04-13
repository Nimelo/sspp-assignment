#include <gtest/gtest.h>
#include <fstream>
#include "CumulativeResults.h"

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  RUN_ALL_TESTS();

  std::ofstream is("cumulative.performance");
  auto result = CumulativeResults::GetInstance();
  is << result;
  is.close();
  return 0;
}
