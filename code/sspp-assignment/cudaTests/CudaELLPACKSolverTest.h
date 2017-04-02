#pragma once

#include "ITest.h"
#include "../cuda/ELLPACKCudaSolver.h"

class CudaELLPACKSolverTest : public ITest {
protected:
  sspp::cuda::ELLPACKCudaSolver<float> *csrParallelSolver;
  virtual void SetUp() {
    csrParallelSolver = new sspp::cuda::ELLPACKCudaSolver<float>();
  }

  virtual void TearDown() {
    delete csrParallelSolver;
  }
};