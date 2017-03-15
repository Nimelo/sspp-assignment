#pragma once
#include "ITest.h"
#include "EllpackSolver.h"
class ELLPACKSolverTest : public ITest {
protected:
  sspp::tools::solvers::ELLPACKSolver *ellpackSolver;
  virtual void SetUp() {
    ellpackSolver = new sspp::tools::solvers::ELLPACKSolver();
  }

  virtual void TearDown() {
    delete ellpackSolver;
  }
};
