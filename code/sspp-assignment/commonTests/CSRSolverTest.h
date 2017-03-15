#pragma once
#include "ITest.h"
#include "CSRSolver.h"
class CSRSolverTest : public ITest {
protected:
  sspp::tools::solvers::CSRSolver *csrSolver;
  virtual void SetUp() {
    csrSolver = new sspp::tools::solvers::CSRSolver();
  }

  virtual void TearDown() {
    delete csrSolver;
  }
};