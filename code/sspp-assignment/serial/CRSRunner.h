#ifndef SSPP_SERIAL_CRSRUNNER_H_
#define SSPP_SERIAL_CRSRUNNER_H_
#include "../common/MetaPerformanceResult.h"
#include "../common/CRS.h"
#include "../common/CRSSolver.h"
#include "../common/Result.h"

namespace sspp {
  namespace serial {
    class CRSRunner {
    public:
      template<typename VALUE_TYPE>
      static common::Result<VALUE_TYPE> run(common::CRS<VALUE_TYPE> & crs, unsigned iterations) {
        double times = 0;
        std::vector<VALUE_TYPE> vector(crs.GetColumns());
        for(unsigned i = 0; i < crs.GetColumns(); ++i) {
          vector[i] = rand() % 100;
        }

        common::CRSSolver<VALUE_TYPE> solver;
        common::Output<VALUE_TYPE> output;
        for(unsigned i = 0; i < iterations; i++) {
          output = solver.Solve(crs, vector);
          times += output.GetMilliseconds();
        }

        double avg_time = static_cast<double>(times) / iterations;

        return common::Result<VALUE_TYPE>(common::MetaPerofmanceResult(crs.GetNonZeros(), iterations, times, 2 * crs.GetNonZeros() / (avg_time == 0 ? 1 : avg_time)),
                                          output);
      };
    };
  }
}
#endif
