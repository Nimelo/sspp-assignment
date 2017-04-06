#pragma once

class SerialParallelComparison {
public:
  SerialParallelComparison(double serial_time,
                           double serial_ops,
                           double parallel_time,
                           double parallel_ops,
                           double speedup,
                           double solution_norm)
    : serial_time_(serial_time),
    serial_ops_(serial_ops),
    parallel_time_(parallel_time),
    parallel_ops_(parallel_ops),
    speedup_(speedup),
    solution_norm_(solution_norm) {
  }

  double GetSerialTime() const {
    return serial_time_;
  }

  double GetParallelTime() const {
    return parallel_time_;
  }

  double GetSerialOps() const {
    return serial_ops_;
  }

  double GetParallelOps() const {
    return parallel_ops_;
  }

  double GetSpeedup() const {
    return speedup_;
  }

  double GetSolutionNorm() const {
    return solution_norm_;
  }
protected:
  double serial_time_;
  double serial_ops_;
  double parallel_time_;
  double parallel_ops_;
  double speedup_;
  double solution_norm_;
};