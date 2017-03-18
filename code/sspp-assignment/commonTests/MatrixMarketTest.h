#pragma once
#include <gtest/gtest.h>
#include "ITest.h"

#include <vector>
#include <sstream>
#include "../common/MatrixMarketEnums.h"
#include "../common/Definitions.h"
#include "MatrixMarket.h"

class MatrixMarketTest : public ITest {
protected:
  std::stringstream ss;
  virtual void SetUp() {
    ss.clear();
  }

  virtual void TearDown() {
    ss.clear();
    ss.seekg(0, ss.beg);
  }

  void WriteHeader(sspp::io::MatrixMarketHeader
                   mmh) {
    using namespace sspp::io;
    ss << mmh.ToString() << "\n";
  }

  void WriteIndices(INDEXING_TYPE rows,
                    INDEXING_TYPE columns,
                    INDEXING_TYPE non_zeros,
                    std::vector<INDEXING_TYPE> & row_indicies,
                    std::vector<INDEXING_TYPE> & column_indicies,
                    std::vector<FLOATING_TYPE> & values) {
    ss << rows << " " << columns << " " << non_zeros << "\n";
    for(int i = 0; i < non_zeros; ++i) {
      ss << row_indicies[i] << " "
        << column_indicies[i] << " "
        << values[i] << "\n";
    }
  }

  void WriteIndicesPattern(INDEXING_TYPE rows,
                           INDEXING_TYPE columns,
                           INDEXING_TYPE non_zeros,
                           std::vector<INDEXING_TYPE> & row_indicies,
                           std::vector<INDEXING_TYPE> & column_indicies) {
    ss << rows << " " << columns << " " << non_zeros << "\n";
    for(int i = 0; i < non_zeros; ++i) {
      ss << row_indicies[i] << " "
        << column_indicies[i] << "\n";
    }
  }
};
