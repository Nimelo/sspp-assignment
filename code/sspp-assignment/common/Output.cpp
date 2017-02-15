#include "Output.h"

representations::output::Output::Output(int size, FLOATING_TYPE * values)
	: N(size), Values(values)
{
}

representations::output::Output::~Output()
{
	delete[] Values;
}
