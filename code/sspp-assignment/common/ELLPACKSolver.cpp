#include "ELLPACKSolver.h"

sspp::representations::Output sspp::tools::solvers::ELLPACKSolver::Solve(sspp::representations::ELLPACK & ellpack, std::vector<FLOATING_TYPE> & b) {
  std::vector<FLOATING_TYPE> x(ellpack.GetRows());

  for(auto i = 0; i < ellpack.GetRows(); i++) {
    FLOATING_TYPE tmp = 0;
    for(auto j = 0; j < ellpack.GetMaxRowNonZeros(); j++) {
      auto index = ellpack.CalculateIndex(i, j);
      tmp += ellpack.GetAS()[index] * b[ellpack.GetJA()[index]];
    }

    x[i] = tmp;
  }

  return sspp::representations::Output(x);
}
