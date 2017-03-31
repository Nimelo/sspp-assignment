#pragma once

#include "ITest.h"
#include "../openmp/CRSOpenMPSolver.h"

class CRSParallelSolverTest : public ITest {
protected:
  sspp::openmp::CRSOpenMPSolver<float> *csrParallelSolver;
  virtual void SetUp() {
    csrParallelSolver = new sspp::openmp::CRSOpenMPSolver<float>();
  }

  virtual void TearDown() {
    delete csrParallelSolver;
  }
};