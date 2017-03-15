#include "Result.h"
#include "Definitions.h"

#include <fstream>
#include <numeric>
#include <cmath>

std::ostream & sspp::representations::result::operator<<(std::ostream & os, const Result & result) {
  os << result.serialResult << LINE_SEPARATOR;
  os << result.parallelResult << LINE_SEPARATOR;
  double diff = 0.0;
  for(int i = 0; i < result.serialResult.output.N; i++)
    diff += fabs(result.parallelResult.output.Values[i] - result.serialResult.output.Values[i]);

  os << diff << LINE_SEPARATOR;

  return os;
}
