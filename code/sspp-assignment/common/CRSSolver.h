#ifndef SSPP_COMMON_CSRSOLVER_H_
#define SSPP_COMMON_CSRSOLVER_H_

#include "../common/CRS.h"
#include "../common/Output.h"
#include "../common/AbstractCRSSolver.h"
#include <chrono>
#include "ChronoStopwatch.h"

namespace sspp {
  namespace common {
    template<typename VALUE_TYPE>
    class CRSSolver : public common::AbstractCRSSolver<VALUE_TYPE> {
    public:
      virtual common::Output<VALUE_TYPE> Solve(common::CRS<VALUE_TYPE> & crs, std::vector<VALUE_TYPE> & vector) {
        std::vector<VALUE_TYPE> x(crs.GetRows());
        static ChronoStopwatch stopwatch_;

        stopwatch_.Start();
        for(unsigned long long i = 0; i < crs.GetRows(); ++i) {
          VALUE_TYPE tmp = 0;
          for(unsigned long long j = crs.GetRowStartIndexes()[i]; j < crs.GetRowStartIndexes()[i + 1]; ++j) {
            tmp += crs.GetValues()[j] * vector[crs.GetColumnIndices()[j]];
          }
          x[i] = tmp;
        }
        stopwatch_.Stop();

        return common::Output<VALUE_TYPE>(x, stopwatch_.GetElapsedSeconds());
      };
    };
  }
}
#endif