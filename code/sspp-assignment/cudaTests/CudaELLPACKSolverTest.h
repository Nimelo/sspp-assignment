#pragma once

#include "ITest.h"
#include "../cuda/ELLPACKCudaSolver.h"

class CudaELLPACKSolverTest : public ITest {
protected:
  sspp::tools::solvers::ELLPACKCudaSolver *csrParallelSolver;
  virtual void SetUp() {
    csrParallelSolver = new sspp::tools::solvers::ELLPACKCudaSolver();
  }

  virtual void TearDown() {
    delete csrParallelSolver;
  }
};