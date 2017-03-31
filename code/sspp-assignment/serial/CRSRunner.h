#ifndef SSPP_SERIAL_CRSRUNNER_H_
#define SSPP_SERIAL_CRSRUNNER_H_
#include "../common/MetaPerformanceResult.h"
#include "../common/CRS.h"
#include "CRSSolver.h"

namespace sspp {
  namespace serial {
    class CRSRunner {
    public:
      template<typename VALUE_TYPE>
      static common::MetaPerofmanceResult run(common::CRS<VALUE_TYPE> & crs, unsigned iterations) {
        unsigned times = 0;
        std::vector<VALUE_TYPE> vector(crs.GetColumns());
        for(unsigned i = 0; i < crs.GetColumns(); ++i) {
          vector[i] = rand() % 100;
        }

        CRSSolver<VALUE_TYPE> solver;
        for(unsigned i = 0; i < iterations; i++) {
          common::Output<VALUE_TYPE> output = solver.Solve(crs, vector);
          times += output.GetMilliseconds();
        }

        double avg_time = static_cast<double>(times) / iterations;

        return common::MetaPerofmanceResult(crs.GetNonZeros(), iterations, avg_time, 2 * crs.GetNonZeros() / (avg_time == 0 ? 1 : avg_time));
      };
    };
  }
}
#endif
