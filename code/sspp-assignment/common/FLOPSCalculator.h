#ifndef SSPP_COMMON_FLOPSCALCULATOR_H_
#define SSPP_COMMON_FLOPSCALCULATOR_H_

namespace sspp {
  namespace tools {
    namespace measurements {
      class FLOPSCalculator {
      public:
        static double calculate(int nz, double miliseconds);
        static double calculate(int nz, long miliseconds);
      };
    }
  }
}

#endif
