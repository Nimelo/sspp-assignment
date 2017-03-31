#pragma once

#include "ITest.h"
#include "../cuda/CRSCudaSolver.h"

class CudaCSRSolverTest : public ITest {
protected:
  sspp::tools::solvers::CRSCudaSolver *csrParallelSolver;
  virtual void SetUp() {
    csrParallelSolver = new sspp::tools::solvers::CRSCudaSolver();
  }

  virtual void TearDown() {
    delete csrParallelSolver;
  }
};