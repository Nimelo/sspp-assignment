#ifndef SSPP_COMMON_CSRSOLVER_H_
#define SSPP_COMMON_CSRSOLVER_H_

#include "CRS.h"
#include "Output.h"
#include "AbstractCRSSolver.h"

namespace sspp {
  namespace common {
    template<typename VALUE_TYPE>
    class CRSSolver : public AbstractCRSSolver<VALUE_TYPE> {
    public:
      virtual Output<VALUE_TYPE> Solve(CRS<VALUE_TYPE> & crs, std::vector<VALUE_TYPE> & vector) {
        std::vector<VALUE_TYPE> x(crs.GetRows());

        for(unsigned i = 0; i < crs.GetRows(); ++i) {
          VALUE_TYPE tmp = 0;
          for(unsigned j = crs.GetRowStartIndexes()[i]; j < crs.GetRowStartIndexes()[i + 1]; ++j) {
            tmp += crs.GetValues()[j] * vector[crs.GetColumnIndices()[j]];
          }
          x[i] = tmp;
        }

        return Output<VALUE_TYPE>(x);
      };
    };
  }
}
#endif