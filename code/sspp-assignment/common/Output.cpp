#include "Output.h"

sspp::representations::Output::Output(std::vector<FLOATING_TYPE>& values)
  :values_{ values } {
}

sspp::representations::Output::Output(const Output & other) {
  Rewrite(*this, other);
}

sspp::representations::Output& sspp::representations::Output::operator=(const Output& rhs) {
  Rewrite(*this, rhs);
  return *this;
}

std::vector<FLOATING_TYPE> sspp::representations::Output::GetValues() const {
  return values_;
}

void sspp::representations::Output::Rewrite(Output& lhs, const Output& rhs) {
  lhs.values_ = std::vector<FLOATING_TYPE>(rhs.values_);
}

std::ostream & sspp::representations::operator<<(std::ostream & os, const sspp::representations::Output & o) {
  for(auto &value : o.values_)
    os << value << SPACE;

  return os;
}
