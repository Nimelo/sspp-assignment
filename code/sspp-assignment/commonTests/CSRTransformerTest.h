#pragma once

#include "ITest.h"
#include "CSRTransformer.h"

class CSRTransformerTest : public ITest {
protected:
  sspp::tools::transformers::CSRTransformer *csrTransformer;
  virtual void SetUp() {
    csrTransformer = new sspp::tools::transformers::CSRTransformer();
  }

  virtual void TearDown() {
    delete csrTransformer;
  }
};