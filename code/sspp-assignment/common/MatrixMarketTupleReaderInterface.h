#ifndef SSPP_COMMON_MATRIXMARKETTUPLEREADERINTERFACE_H_
#define SSPP_COMMON_MATRIXMARKETTUPLEREADERINTERFACE_H_

#include "MatrixMarketTuple.h"
#include <string>

namespace sspp {
  namespace common {
    template<typename VALUE_TYPE>
    class MatrixMarketTupleReaderInterface {
    public:
      virtual ~MatrixMarketTupleReaderInterface() {};
      virtual MatrixMarketTuple<VALUE_TYPE> Get(std::string line, bool is_pattern) = 0;
    public:
      virtual VALUE_TYPE GetPatternValue() const = 0;
    };
  }
}
#endif
