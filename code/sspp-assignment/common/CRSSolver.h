#ifndef SSPP_COMMON_CSRSOLVER_H_
#define SSPP_COMMON_CSRSOLVER_H_

#include "../common/CRS.h"
#include "../common/Output.h"
#include "../common/AbstractCRSSolver.h"
#include <chrono>

namespace sspp {
  namespace common {
    template<typename VALUE_TYPE>
    class CRSSolver : public common::AbstractCRSSolver<VALUE_TYPE> {
    public:
      virtual common::Output<VALUE_TYPE> Solve(common::CRS<VALUE_TYPE> & crs, std::vector<VALUE_TYPE> & vector) {
        std::vector<VALUE_TYPE> x(crs.GetRows());

        std::chrono::steady_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        for(unsigned i = 0; i < crs.GetRows(); ++i) {
          VALUE_TYPE tmp = 0;
          for(unsigned j = crs.GetRowStartIndexes()[i]; j < crs.GetRowStartIndexes()[i + 1]; ++j) {
            tmp += crs.GetValues()[j] * vector[crs.GetColumnIndices()[j]];
          }
          x[i] = tmp;
        }
        std::chrono::steady_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        return common::Output<VALUE_TYPE>(x, std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000000.0);
      };
    };
  }
}
#endif