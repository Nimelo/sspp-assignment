#pragma once

class OpenMPPerformance {
public:
  OpenMPPerformance(unsigned long long threads_number, double speedup)
    : threads_number_(threads_number),
    speedup_(speedup) {
  }

  unsigned long long GetThreadsNumber() const {
    return threads_number_;
  }

  double GetSpeedup() const {
    return speedup_;
  }

protected:
  unsigned long long threads_number_;
  double speedup_;
};