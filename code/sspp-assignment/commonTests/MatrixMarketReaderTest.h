#pragma once
#include "ITest.h"
#include "../common/MatrixMarketReader.h"

class MatrixMarketReaderTest : public ITest {
protected:
  sspp::io::readers::MatrixMarketReader *matrixMarketReader;
  virtual void SetUp() {
    matrixMarketReader = new sspp::io::readers::MatrixMarketReader();
  }

  virtual void TearDown() {
    delete matrixMarketReader;
  }
};