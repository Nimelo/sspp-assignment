#include "SingleHeader.h"
#include "Definitions.h"

#include <numeric>
#include <fstream>

std::ostream & sspp::representations::result::single::operator<<(std::ostream & os, const SingleResult & result) {
  os << static_cast<double>(std::accumulate(result.execution_times_.begin(), result.execution_times_.end(), 0)) / result.execution_times_.size() << LINE_SEPARATOR;
  for(int i = 0; i < result.execution_times_.size(); i++)
    os << result.execution_times_[i] << SPACE;
  os << LINE_SEPARATOR;
  os << result.output_;
  return os;
}

sspp::representations::Output sspp::representations::result::single::SingleResult::GetOutput() const {
  return this->output_;
}

std::vector<double> sspp::representations::result::single::SingleResult::GetExecutionTimes() const {
  return this->execution_times_;
}
