#include "Output.h"

sspp::representations::Output::Output() {
}

sspp::representations::Output::Output(int size, FLOATING_TYPE * values)
  : Values(values), N(size) {
}

sspp::representations::Output::Output(const Output & other) {
  N = other.N;
  Values = new FLOATING_TYPE[N];
  for(auto i = 0; i < N; i++)
    Values[i] = other.Values[i];
}

sspp::representations::Output & sspp::representations::Output::operator=(representations::Output other) {
  N = other.N;
  Values = new FLOATING_TYPE[N];
  for(auto i = 0; i < N; i++)
    Values[i] = other.Values[i];

  return *this;
}

sspp::representations::Output::~Output() {
  if(Values != 0)
    delete[] Values;
}

std::ostream & sspp::representations::operator<<(std::ostream & os, const sspp::representations::Output & o) {
  for(auto i = 0; i < o.N; i++)
    os << o.Values[i] << SPACE;

  return os;
}
