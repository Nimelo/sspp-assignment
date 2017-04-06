#pragma once

class OpenMPPerformance {
public:
  OpenMPPerformance(unsigned threads_number, double speedup)
    : threads_number_(threads_number),
    speedup_(speedup) {
  }

  unsigned GetThreadsNumber() const {
    return threads_number_;
  }

  double GetSpeedup() const {
    return speedup_;
  }

protected:
  unsigned threads_number_;
  double speedup_;
};