#include "Result.h"
#include "Definitions.h"

#include <fstream>
#include <numeric>
#include <cmath>

std::ostream & sspp::representations::result::operator<<(std::ostream & os, const Result & result) {
  os << result.serial_result_ << LINE_SEPARATOR;
  os << result.parallel_result_ << LINE_SEPARATOR;
  double diff = 0.0;
  for(int i = 0; i < result.serial_result_.GetOutput().GetValues().size(); i++)
    diff += fabs(result.parallel_result_.GetOutput().GetValues()[i] - result.serial_result_.GetOutput().GetValues()[i]);

  os << diff << LINE_SEPARATOR;

  return os;
}

sspp::representations::result::single::SingleResult sspp::representations::result::Result::GetSerial() const {
  return this->serial_result_;
}

sspp::representations::result::single::SingleResult sspp::representations::result::Result::GetParallel() const {
  return  this->parallel_result_;
}
