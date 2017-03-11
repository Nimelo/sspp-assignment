#include "Result.h"

#include <fstream>
#include <numeric>

std::ostream & representations::result::operator<<(std::ostream & os, const Result & o)
{
	os << o.output;
	return os;
}

void representations::result::Result::save(std::string metadata, std::string output)
{
	std::fstream stream;

	stream.open(metadata, std::fstream::out | std::fstream::trunc);
	stream << (double)std::accumulate(this->executionTimes.begin(), this->executionTimes.end(), 0) / this->executionTimes.size() << std::endl;
	for (int i = 0; i < this->executionTimes.size(); i++)
		stream << this->executionTimes[i] << " ";
	stream.close();

	stream.open(output, std::fstream::out | std::fstream::trunc);
	stream << this->output;
	stream.close();
}
