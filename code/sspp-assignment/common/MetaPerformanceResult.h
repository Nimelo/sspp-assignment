#ifndef SSPP_COMMON_METAPERFORMANCERESULT_H_
#define SSPP_COMMON_METAPERFORMANCERESULT_H_
#include <ostream>

namespace sspp {
  namespace common {
    struct MetaPerofmanceResult {
      MetaPerofmanceResult(unsigned long long non_zeros,
                           unsigned long long iterations,
                           double averaget_time,
                           double flops) :
        NonZeros(non_zeros),
        Iterations(iterations),
        Time(averaget_time),
        Flops(flops) {
      };
      unsigned long long NonZeros;
      unsigned long long Iterations;
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
