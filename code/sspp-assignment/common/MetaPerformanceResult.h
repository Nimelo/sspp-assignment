#ifndef SSPP_COMMON_METAPERFORMANCERESULT_H_
#define SSPP_COMMON_METAPERFORMANCERESULT_H_
#include <ostream>

namespace sspp {
  namespace common {
    struct MetaPerofmanceResult {
      MetaPerofmanceResult(unsigned non_zeros,
                           unsigned iterations,
                           double averaget_time,
                           double flops) :
        NonZeros(non_zeros),
        Iterations(iterations),
        Time(averaget_time),
        Flops(flops) {
      };
      unsigned NonZeros;
      unsigned Iterations;
      double Time;
      double Flops;

      friend std::ostream& operator <<(std::ostream& os, const MetaPerofmanceResult& result) {
        os << "NonZeros: " << result.NonZeros << std::endl;
        os << "Iterations: " << result.Iterations << std::endl;
        os << "Time: " << result.Time << std::endl;
        os << "Flops: " << result.Flops << std::endl;
        return os;
      };
    };
  }
}

#endif
