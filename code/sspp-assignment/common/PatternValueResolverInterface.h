#ifndef SSPP_COMMON_PATTERNVALUERESOLVERINTERFACE_H_
#define SSPP_COMMON_PATTERNVALUERESOLVERINTERFACE_H_

#include "MatrixMarketTuple.h"
#include <string>

namespace sspp {
  namespace common {
    template<typename VALUE_TYPE>
    class PatternValueResolverInterface {
    public:
      virtual ~PatternValueResolverInterface() {};
      virtual VALUE_TYPE GetPatternValue() const = 0;
    };
  }
}
#endif
