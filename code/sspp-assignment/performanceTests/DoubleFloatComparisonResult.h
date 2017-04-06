#pragma once

class DoubleFloatComparison {
public:
  DoubleFloatComparison(double float_time,
                        double double_time,
                        double flops,
                        double dp_flops,
                        double result_norm) :
    float_time_(float_time),
    double_time_(double_time),
    flops_(flops),
    dp_flops_(dp_flops),
    result_norm_(result_norm) {
  }

  double GetFloatTime() const {
    return float_time_;
  }

  double GetDoubleTime() const {
    return double_time_;
  }

  double GetFlops() const {
    return flops_;
  }

  double GetDPFlops() const {
    return dp_flops_;
  }

  double GetResultNorm() const {
    return result_norm_;
  }

  friend std::ostream & operator << (std::ostream & os,
                                     const DoubleFloatComparison & result) {
    os << "Float: " << result.float_time_ << "s\n"
      << "Double: " << result.double_time_ << "s\n"
      << "FLOPS: " << result.flops_ << "\n"
      << "DP FLOPS: " << result.dp_flops_ << "\n"
      << "Float/double ratio: " << result.float_time_ / result.double_time_ << "\n"
      << "Solution norm: " << result.result_norm_;
    return os;
  }
protected:
  double float_time_;
  double double_time_;
  double flops_;
  double dp_flops_;
  double result_norm_;
};
