#include "SingleHeader.h"
#include "Definitions.h"

#include <numeric>
#include <fstream>

std::ostream & representations::result::single::operator<<(std::ostream & os, const SingleResult & result)
{
	os << (double)std::accumulate(result.executionTimes.begin(), result.executionTimes.end(), 0) / result.executionTimes.size() << LINE_SEPARATOR;
	for (int i = 0; i < result.executionTimes.size(); i++)
		os << result.executionTimes[i] << SPACE;
	os << LINE_SEPARATOR;
	os << result.output;
	return os;
}
