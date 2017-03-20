#include "ELLPACKOpenMPSolver.h"
#include <omp.h>

sspp::representations::Output *sspp::tools::solvers::ELLPACKOpenMPSolver::Solve(representations::ELLPACK & ellpack, std::vector<FLOATING_TYPE> & b) {
  auto x = new std::vector<FLOATING_TYPE>(ellpack.GetRows());
#pragma omp for
  for(auto i = 0; i < ellpack.GetRows(); i++)
    x->at(i) = 0.0;

#pragma omp parallel shared(ellpack, b, x)
  for(auto i = 0; i < ellpack.GetRows(); i++) {
    FLOATING_TYPE tmp = 0;
    for(auto j = 0; j < ellpack.GetMaxRowNonZeros(); j++) {
      auto index = ellpack.CalculateIndex(i, j);
      tmp += ellpack.GetAS()->at(index) * b[ellpack.GetJA()->at(index)];
    }
    x->at(i) = tmp;
  }

  //TODO: examine omp barriers
#pragma omp barrier
  return new representations::Output(x);
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
