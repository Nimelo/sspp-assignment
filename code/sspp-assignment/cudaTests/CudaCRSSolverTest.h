#pragma once

#include "ITest.h"
#include "../cuda/CRSCudaSolver.h"

class CudaCSRSolverTest : public ITest {
protected:
  sspp::cuda::CRSCudaSolver<float> *csrParallelSolver;
  virtual void SetUp() {
    csrParallelSolver = new sspp::cuda::CRSCudaSolver<float>();
  }

  virtual void TearDown() {
    delete csrParallelSolver;
  }
};