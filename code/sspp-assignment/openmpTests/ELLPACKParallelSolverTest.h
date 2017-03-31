#pragma once

#include "ITest.h"
#include "../openmp/ELLPACKOpenMPSolver.h"

class ELLPACKParallelSolverTest : public ITest {
protected:
  sspp::openmp::ELLPACKOpenMPSolver<float> *ellpackParallelSolver;
  virtual void SetUp() {
    ellpackParallelSolver = new sspp::openmp::ELLPACKOpenMPSolver<float>();
  }

  virtual void TearDown() {
    delete ellpackParallelSolver;
  }
};