#ifndef SSPP_SERIAL_ELLPACKRUNNER_H_
#define SSPP_SERIAL_ELLPACKRUNNER_H_
#include "../common/MetaPerformanceResult.h"
#include "../common/ELLPACK.h"
#include "../common/EllpackSolver.h"


namespace sspp {
  namespace serial {
    class ELLPACKRunner {
    public:
      template<typename VALUE_TYPE>
      static common::MetaPerofmanceResult run(common::ELLPACK<VALUE_TYPE> & crs, unsigned iterations) {
        unsigned times = 0;
        std::vector<VALUE_TYPE> vector(crs.GetColumns());
        for(unsigned i = 0; i < crs.GetColumns(); ++i) {
          vector[i] = rand() % 100;
        }
        
        common::ELLPACKSolver<VALUE_TYPE> solver;
        for(unsigned i = 0; i < iterations; i++) {
          common::Output<VALUE_TYPE> output = solver.Solve(crs, vector);
          times += output.GetSeconds();
        }

        double avg_time = static_cast<double>(times) / iterations;

        return common::MetaPerofmanceResult(crs.GetNonZeros(), iterations, avg_time, 2 * crs.GetNonZeros() / (avg_time == 0 ? 1 : avg_time));
      };
    };
  }
}
#endif
