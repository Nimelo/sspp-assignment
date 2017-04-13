#pragma once

#include "SerialParallelComparison.h"
#include <ostream>

class CudaCumulativeResult {
public:
  CudaCumulativeResult(const CudaCumulativeResult & other) :
    crs_float_(other.crs_float_),
    ellpack_float_(other.ellpack_float_),
    crs_double_(other.crs_double_),
    ellpack_double_(other.ellpack_double_) {
  }

  CudaCumulativeResult(SerialParallelComparison crs_float,
                       SerialParallelComparison crs_double,
                       SerialParallelComparison ellpack_float,
                       SerialParallelComparison ellpack_double)
    : crs_float_{ crs_float },
    ellpack_float_{ ellpack_float },
    crs_double_{ crs_double },
    ellpack_double_{ ellpack_double } {
  }
  friend std::ostream & operator << (std::ostream & os, const CudaCumulativeResult & result) {

    os << "no" << ' ' << "crs_float" << ' ' << "crs_double" << ' ' << "ellpack_float" << ' ' << "ellpack_double" << std::endl;
    os << 1 << ' ' << result.crs_float_.GetParallelOps() << ' ';
    os << result.crs_double_.GetParallelOps() << ' ';
    os << result.ellpack_float_.GetParallelOps() << ' ';
    os << result.ellpack_double_.GetParallelOps() << std::endl;

    return os;
  }
public:
  SerialParallelComparison crs_float_;
  SerialParallelComparison ellpack_float_;
  SerialParallelComparison crs_double_;
  SerialParallelComparison ellpack_double_;
};
