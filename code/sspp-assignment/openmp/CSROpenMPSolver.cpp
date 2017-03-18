#include "CSROpenMPSolver.h"
#include <omp.h>

sspp::representations::Output sspp::tools::solvers::CSROpenMPSolver::Solve(representations::CSR & csr, std::vector<FLOATING_TYPE> & b) {
  std::vector<FLOATING_TYPE> x(csr.GetRows());

#pragma omp for
  for(auto i = 0; i < csr.GetRows(); i++)
    x[i] = 0.0;

#pragma omp parallel shared(csr, b, x)
  {
    int threads = omp_get_num_threads(),
      threadId = omp_get_thread_num();
    int lowerBoundary = csr.GetRows() * threadId / threads,
      upperBoundary = csr.GetRows() *(threadId + 1) / threads;

    //#pragma ivdep
    for(auto i = lowerBoundary; i < upperBoundary; i++) {
      for(auto j = csr.GetIRP()[i]; j < csr.GetIRP()[i + 1]; j++) {
        x[i] += csr.GetAS()[j] * b[csr.GetJA()[j]];
      }
    }
  }
  //TODO: examine omp barriers
#pragma omp barrier

  return representations::Output(x);
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
