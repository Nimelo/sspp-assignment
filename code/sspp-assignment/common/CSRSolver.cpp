#include "CSRSolver.h"

sspp::representations::Output sspp::tools::solvers::CSRSolver::Solve(sspp::representations::CSR &csr, std::vector<FLOATING_TYPE> & b) {
  std::vector<FLOATING_TYPE> x(csr.GetRows());

  for(auto i = 0; i < csr.GetRows(); i++) {
    FLOATING_TYPE tmp = 0;
    for(auto j = csr.GetIRP()[i]; j < csr.GetIRP()[i + 1]; j++) {
      tmp += csr.GetAS()[j] * b[csr.GetJA()[j]];
    }
    x[i] = tmp;
  }

  return sspp::representations::Output(x);
}
