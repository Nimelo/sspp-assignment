#include "CSRSolver.h"

sspp::representations::Output sspp::tools::solvers::CSRSolver::solve(sspp::representations::CSR &csr, FLOATING_TYPE * b) {
  FLOATING_TYPE *x = new FLOATING_TYPE[csr.M];

  for(auto i = 0; i < csr.M; i++) {
    FLOATING_TYPE tmp = 0;

    for(auto j = csr.IRP[i]; j < csr.IRP[i + 1]; j++) {
      tmp += csr.AS[j] * b[csr.JA[j]];
    }

    x[i] = tmp;
  }

  return representations::Output(csr.M, x);
}
