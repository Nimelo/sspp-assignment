#ifndef SSPP_COMMON_CHRONOSTOPWATCH_H_
#define SSPP_COMMON_CHRONOSTOPWATCH_H_

#include "AbstractStopwatch.h"
#include <chrono>

namespace sspp {
  namespace common {
    class ChronoStopwatch : public AbstractStopWatch {
    public:
      void Start() override {
        begin_ = std::chrono::high_resolution_clock::now();
      }

      void Stop() override {
        end_ = std::chrono::high_resolution_clock::now();
      }
      void Reset() override {

      }
      double GetElapsedSeconds() const override {
        auto elapsed_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end_ - begin_).count();
        return elapsed_microseconds * 0.001 * 0.001;
      }
    protected:
      std::chrono::steady_clock::time_point begin_;
      std::chrono::steady_clock::time_point end_;
    };
  }
}
#endif
