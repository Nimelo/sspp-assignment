#pragma once

#include "ITest.h"
#include "../cuda/CSRCudaSolver.h"

class CudaCSRSolverTest : public ITest {
protected:
  sspp::tools::solvers::CSRCudaSolver *csrParallelSolver;
  virtual void SetUp() {
    csrParallelSolver = new sspp::tools::solvers::CSRCudaSolver();
  }

  virtual void TearDown() {
    delete csrParallelSolver;
  }
};