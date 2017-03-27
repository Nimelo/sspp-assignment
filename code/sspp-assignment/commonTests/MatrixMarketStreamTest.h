#pragma once
#include <gtest/gtest.h>
#include "ITest.h"

#include <vector>
#include <sstream>
#include "MatrixMarketStream.h"

class MatrixMarketStreamTest : public ITest {
protected:
  std::stringstream ss;
  virtual void SetUp() {
    ss.clear();
  }

  virtual void TearDown() {
    ss.clear();
    ss.seekg(0, ss.beg);
  }

  void WriteHeader(sspp::common::MatrixMarketHeader & mmh) {
    ss << mmh.ToString() << std::endl;
  }

  template<typename MAGNITUDE_TYPE, typename INDICES_TYPE, typename VALUE_TYPE>
  void WriteIndices(MAGNITUDE_TYPE rows,
                    MAGNITUDE_TYPE columns,
                    MAGNITUDE_TYPE non_zeros,
                    std::vector<INDICES_TYPE> & row_indicies,
                    std::vector<INDICES_TYPE> & column_indicies,
                    std::vector<VALUE_TYPE> & values) {
    ss << rows << " " << columns << " " << non_zeros << std::endl;
    for(unsigned i = 0; i < non_zeros; ++i) {
      ss << row_indicies[i] << " "
        << column_indicies[i] << " "
        << values[i] << std::endl;
    }
  }

  template<typename MAGNITUDE_TYPE, typename INDICES_TYPE>
  void WriteIndicesPattern(MAGNITUDE_TYPE rows,
                           MAGNITUDE_TYPE columns,
                           MAGNITUDE_TYPE non_zeros,
                           std::vector<INDICES_TYPE> & row_indicies,
                           std::vector<INDICES_TYPE> & column_indicies) {
    ss << rows << " " << columns << " " << non_zeros << std::endl;
    for(unsigned i = 0; i < non_zeros; ++i) {
      ss << row_indicies[i] << " "
        << column_indicies[i] << std::endl;
    }
  }
};
