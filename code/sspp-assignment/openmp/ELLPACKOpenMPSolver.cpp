#include "ELLPACKOpenMPSolver.h"
#include <omp.h>

sspp::representations::Output sspp::tools::solvers::ELLPACKOpenMPSolver::solve(representations::ELLPACK & ellpack, FLOATING_TYPE * b) {
  FLOATING_TYPE *x = new FLOATING_TYPE[ellpack.M];
#pragma omp for
  for(auto i = 0; i < ellpack.M; i++)
    x[i] = 0.0;

#pragma omp parallel shared(ellpack, b, x)
  for(auto i = 0; i < ellpack.M; i++) {
    FLOATING_TYPE tmp = 0;
    for(auto j = 0; j < ellpack.MAXNZ; j++) {
      tmp += ellpack.AS[i][j] * b[ellpack.JA[i][j]];
    }

    x[i] = tmp;
  }

  //TODO: examine omp barriers
#pragma omp barrier
  return representations::Output(ellpack.M, x);
}

sspp::tools::solvers::ELLPACKOpenMPSolver::ELLPACKOpenMPSolver(int threads)
  :threads(threads) {
  omp_set_dynamic(0);
  omp_set_num_threads(threads);
}

void sspp::tools::solvers::ELLPACKOpenMPSolver::setThreads(int threads) {
  this->threads = threads;
  omp_set_dynamic(0);
  omp_set_num_threads(threads);
}
