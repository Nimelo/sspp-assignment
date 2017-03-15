#pragma once

#include "ITest.h"
#include "ELLPACKTransformer.h"

class ELLPACKTransformerTest : public ITest {
protected:
  sspp::tools::transformers::ELLPACKTransformer *ellpackTransformer;
  virtual void SetUp() {
    ellpackTransformer = new sspp::tools::transformers::ELLPACKTransformer();
  }

  virtual void TearDown() {
    delete ellpackTransformer;
  }
};
