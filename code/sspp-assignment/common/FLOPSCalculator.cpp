#include "FLOPSCalculator.h"

double tools::measurements::calculators::FLOPSCalculator::calculate(int nz, double miliseconds)
{
	return 2 * nz / (miliseconds / 1000);
}

double tools::measurements::calculators::FLOPSCalculator::calculate(int nz, long miliseconds)
{
	return 2 * nz / (static_cast<double>(miliseconds) / 1000);
}
