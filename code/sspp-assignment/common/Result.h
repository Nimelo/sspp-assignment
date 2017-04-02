#ifndef SSPP_COMMON_RESULT_H_
#define SSPP_COMMON_RESULT_H_

#include "Output.h"

#include <ostream>
#include <string>
#include <vector>
#include "MetaPerformanceResult.h"

namespace sspp {
  namespace common {
    template<typename VALUE_TYPE>
    struct Result {
      Result(MetaPerofmanceResult & meta, Output<VALUE_TYPE> output) :
        meta(meta), output(output) {
      }
      MetaPerofmanceResult meta;
      Output<VALUE_TYPE> output;

      Result(const Result<VALUE_TYPE> & other) {
        Swap(*this, other);
      }
    protected:
      static void Swap(Result<VALUE_TYPE> & lhs, Result<VALUE_TYPE> & rhs) {
        lhs.meta = rhs.meta;
        lhs.output = rhs.output;
      }
    };
  }
}


#endif
