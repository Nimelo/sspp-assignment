#include "Output.h"

sspp::representations::Output::Output(std::vector<FLOATING_TYPE> * values)
  :values_{ values } {
}

sspp::representations::Output::Output()
  : values_{ nullptr } {
}

sspp::representations::Output::~Output() {
  if(values_ != nullptr)
    delete values_;
}

std::vector<FLOATING_TYPE> * sspp::representations::Output::GetValues() const {
  return values_;
}

std::ostream & sspp::representations::operator<<(std::ostream & os, const sspp::representations::Output & o) {
  for(auto &value : *o.values_)
    os << value << SPACE;

  return os;
}
