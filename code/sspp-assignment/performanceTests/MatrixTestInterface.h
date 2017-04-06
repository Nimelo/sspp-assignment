#pragma once

#include <gtest/gtest.h>
#include "ELLPACK.h"
#include "CRS.h"
#include "MarketMatrixReader.h"
#include "FloatPatternResolver.h"
#include <fstream>
#include "ChronoStopwatch.h"
#include "ELLPACKTransformer.h"
#include "CRSTransformer.h"
#include "TestingUtilities.h"
#include "MatrixContainer.h"

class MatrixTestInterface : public ::testing::Test {
public:
  void SetUp() {
    using namespace sspp::common;
    static_key_ = GetKey();
    if(MatrixContainer::GetInstance().Exist(GetKey())) {
      TEST_COUT << "Matrix with key: " << GetKey() << " already loaded." << std::endl;
    } else {
      TEST_COUT << "Reading matrix-market from: " << GetPath() << std::endl;
      stopwatch_.Start();
      MatrixContainer::GetInstance().LoadOnce(GetPath(), GetKey());
      stopwatch_.Stop();
      TEST_COUT << "Finished in: " << stopwatch_.GetElapsedSeconds() << "s" << std::endl;
    }
  }

  static void TearDownTestCase() {
    TEST_COUT << "Deleting matrix with key: " << static_key_ << std::endl;
    MatrixContainer::GetInstance().Delete(static_key_);
  }
protected:
  template<typename T>
  sspp::common::CRS<T> GetCRS() {
    TEST_COUT << "Transforming to CRS<T>: " << GetKey() << std::endl;
    stopwatch_.Start();
    auto crs = MatrixContainer::GetInstance().GetCRS<T>(GetKey());
    stopwatch_.Stop();
    TEST_COUT << "Finished in: " << stopwatch_.GetElapsedSeconds() << "s" << std::endl;
    return crs;
  }

  template<typename T>
  sspp::common::ELLPACK<T> GetELLPACK() {
    TEST_COUT << "Transforming to ELLPACK<T>: " << GetKey() << std::endl;
    stopwatch_.Start();
    auto ellpack = MatrixContainer::GetInstance().GetELLPACK<T>(GetKey());
    stopwatch_.Stop();
    TEST_COUT << "Finished in: " << stopwatch_.GetElapsedSeconds() << "s" << std::endl;
    return ellpack;
  }
  virtual std::string GetPath() = 0;
  virtual std::string GetKey() = 0;

  sspp::common::ChronoStopwatch stopwatch_;
  static std::string static_key_;
};
