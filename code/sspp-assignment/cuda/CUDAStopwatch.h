#ifndef SSPP_CUDA_CUDASTOPWATCH_H_
#define SSPP_CUDA_CUDASTOPWATCH_H_
#include "../common/AbstractStopwatch.h"
#include <windows.h>

namespace sspp {
  namespace cuda {
    class CUDAStopwatch : public common::AbstractStopWatch<LARGE_INTEGER> {
    public:
      CUDAStopwatch() {
        LARGE_INTEGER temp;
        QueryPerformanceFrequency(&temp);
        frequency_ = static_cast<double>(temp.QuadPart) / 1000.0;
      }
      void Start() override {
        QueryPerformanceCounter(&begin_);
      }
      void Stop() override {
        QueryPerformanceCounter(&end_);
      }
      void Reset() override {
      }
      double GetElapsedSeconds() const override {
        auto elapsed_time = (static_cast<double>(end_.QuadPart) - static_cast<double>(begin_.QuadPart)) / frequency_;
        auto elapsed_time_in_magnitude = elapsed_time * 0.001;
        return elapsed_time_in_magnitude;
      }
    protected:
      double frequency_;
    };
  }
}

#endif
