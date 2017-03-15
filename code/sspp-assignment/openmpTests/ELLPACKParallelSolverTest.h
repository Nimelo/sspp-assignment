#pragma once

#include "ITest.h"
#include "../openmp/ELLPACKOpenMPSolver.h"

class ELLPACKParallelSolverTest : public ITest {
protected:
  sspp::tools::solvers::ELLPACKOpenMPSolver *ellpackParallelSolver;
  virtual void SetUp() {
    ellpackParallelSolver = new sspp::tools::solvers::ELLPACKOpenMPSolver();
  }

  virtual void TearDown() {
    delete ellpackParallelSolver;
  }
};