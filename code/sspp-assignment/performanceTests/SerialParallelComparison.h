#pragma once

template<typename T>
class SerialParallelComparison {
public:
  SerialParallelComparison(T serial_time,
                           T serial_ops,
                           T parallel_time,
                           T parallel_ops,
                           double speedup,
                           double solution_norm)
    : serial_time_(serial_time),
    serial_ops_(serial_ops),
    parallel_time_(parallel_time),
    parallel_ops_(parallel_ops),
    speedup_(speedup),
    solution_norm_(solution_norm) {
  }

  T GetSerialTime() const {
    return serial_time_;
  }

  T GetParallelTime() const {
    return parallel_time_;
  }

  T GetSerialOps() const {
    return serial_ops_;
  }

  T GetParallelOps() const {
    return parallel_ops_;
  }

  double GetSpeedup() const {
    return speedup_;
  }

  double GetSolutionNorm() const {
    return solution_norm_;
  }
protected:
  T serial_time_;
  T serial_ops_;
  T parallel_time_;
  T parallel_ops_;
  double speedup_;
  double solution_norm_;
};