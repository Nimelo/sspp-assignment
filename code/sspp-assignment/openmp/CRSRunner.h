#ifndef SSPP_OPENMP_CRSRUNNER_H_
#define SSPP_OPENMP_CRSRUNNER_H_
#include "../common/MetaPerformanceResult.h"
#include "../common/CRS.h"
#include "CRSOpenMPSolver.h"
#include "../common/Result.h"

namespace sspp {
  namespace openmp {
    class CRSRunner {
    public:
      template<typename VALUE_TYPE>
      static common::Result<VALUE_TYPE> run(common::CRS<VALUE_TYPE> & crs, unsigned iterations, unsigned threads) {
        double times = 0;
        std::vector<VALUE_TYPE> vector(crs.GetColumns());
        for(unsigned i = 0; i < crs.GetColumns(); ++i) {
          vector[i] = rand() % 100;
        }

        CRSOpenMPSolver<VALUE_TYPE> solver;
        solver.SetThreads(threads);
        common::Output<VALUE_TYPE> output;
        for(unsigned i = 0; i < iterations; i++) {
          output = solver.Solve(crs, vector);
          times += output.GetSeconds();
        }

        double avg_time = static_cast<double>(times) / iterations;


        return common::Result<VALUE_TYPE>(common::MetaPerofmanceResult(crs.GetNonZeros(), iterations, times, 2 * crs.GetNonZeros() / (avg_time == 0 ? 1 : avg_time)),
                                          output);
      };
    };
  }
}
#endif
