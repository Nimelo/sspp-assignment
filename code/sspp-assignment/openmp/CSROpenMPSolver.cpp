#include "CSROpenMPSolver.h"
#include <omp.h>

sspp::representations::Output *sspp::tools::solvers::CSROpenMPSolver::Solve(representations::CSR & csr, std::vector<FLOATING_TYPE> & b) {
  auto x = new std::vector<FLOATING_TYPE>(csr.GetRows());

#pragma omp for
  for(auto i = 0; i < csr.GetRows(); i++)
    x->at(i) = 0.0;

#pragma omp parallel shared(csr, b, x)
  {
    int threads = omp_get_num_threads(),
      threadId = omp_get_thread_num();
    int lowerBoundary = csr.GetRows() * threadId / threads,
      upperBoundary = csr.GetRows() *(threadId + 1) / threads;

    //#pragma ivdep
    for(auto i = lowerBoundary; i < upperBoundary; i++) {
      for(auto j = csr.GetIRP()->at(i); j < csr.GetIRP()->at(i + 1); j++) {
        x->at(i) += csr.GetAS()->at(j) * b[csr.GetJA()->at(j)];
      }
    }
  }
  //TODO: examine omp barriers
#pragma omp barrier

  return new representations::Output(x);
}

sspp::tools::solvers::CSROpenMPSolver::CSROpenMPSolver(int threads)
  :threads(threads) {
  omp_set_dynamic(0);
  omp_set_num_threads(threads);
}

void sspp::tools::solvers::CSROpenMPSolver::setThreads(int threads) {
  this->threads = threads;
  omp_set_dynamic(0);
  omp_set_num_threads(threads);
}
