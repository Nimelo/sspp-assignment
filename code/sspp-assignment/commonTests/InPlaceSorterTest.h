#pragma once
#include "ITest.h"
#include "StableSorter.h"
class InPlaceSorterTest : public ITest {
protected:
  sspp::common::StableSorter *sorter;
  virtual void SetUp() {
    sorter = new sspp::common::StableSorter();
  }

  virtual void TearDown() {
    delete sorter;
  }
};