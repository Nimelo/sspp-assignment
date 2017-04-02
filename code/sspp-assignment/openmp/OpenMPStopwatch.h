#ifndef SSPP_OPENMP_OPENMPSTOPWATCH_H_
#define SSPP_OPENMP_OPENMPSTOPWATCH_H_
#include "../common/AbstractStopwatch.h"

#include <omp.h>

namespace sspp {
  namespace openmp {
    class OpenMPStopwatch : public common::AbstractStopWatch<double> {
    public:
      void Start() override {
        begin_ = omp_get_wtime();
      }

      void Stop() override {
        end_ = omp_get_wtime();
      }

      void Reset() override {
      }

      double GetElapsedSeconds() const override {
        return end_ - begin_;
      }
    };
  }
}

#endif
