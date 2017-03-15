#include "FLOPSCalculator.h"

double sspp::tools::measurements::FLOPSCalculator::calculate(int nz, double miliseconds)
{
	return 2 * nz / (miliseconds / 1000);
}

double sspp::tools::measurements::FLOPSCalculator::calculate(int nz, long miliseconds)
{
	return 2 * nz / (static_cast<double>(miliseconds) / 1000);
}
