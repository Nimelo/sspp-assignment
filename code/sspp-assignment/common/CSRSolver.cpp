#include "CSRSolver.h"

sspp::representations::Output * sspp::tools::solvers::CSRSolver::Solve(sspp::representations::CSR &csr, std::vector<FLOATING_TYPE> & b) {
  auto x = new std::vector<FLOATING_TYPE>(csr.GetRows());

  for(auto i = 0; i < csr.GetRows(); i++) {
    FLOATING_TYPE tmp = 0;
    for(auto j = csr.GetIRP()->at(i); j < csr.GetIRP()->at(i + 1); j++) {
      tmp += csr.GetAS()->at(j) * b[csr.GetJA()->at(j)];
    }
    x->at(i) = tmp;
  }

  return new sspp::representations::Output(x);
}
