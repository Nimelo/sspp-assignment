#ifndef SSPP_COMMON_DOUBLEPATTERNRESOLVER_H_
#define SSPP_COMMON_DOUBLEPATTERNRESOLVER_H_


#include <string>
#include "PatternValueResolverInterface.h"

namespace sspp {
  namespace common {
    class DoublePatternResolver : public PatternValueResolverInterface<double> {
    public:
      double GetPatternValue() const {
        return static_cast<double>(rand() % 100);
      }
    };
  }
}
#endif
