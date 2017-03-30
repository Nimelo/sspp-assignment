#ifndef SSPP_COMMON_FLOATPATTERNRESOLVER_H_
#define SSPP_COMMON_FLOATPATTERNRESOLVER_H_


#include <string>
#include "PatternValueResolverInterface.h"

namespace sspp {
  namespace common {
    class FloatPatternResolver : public PatternValueResolverInterface<float> {
    public:
      float GetPatternValue() const {
        return static_cast<float>(rand() % 100);
      }
    };
  }
}
#endif
