#include "Output.h"

representations::output::Output::Output(int size, FLOATING_TYPE * values)
	: size(size), values(values)
{
}

representations::output::Output::~Output()
{
	delete[] values;
}

int representations::output::Output::getSize() const
{
	return size;
}

FLOATING_TYPE * representations::output::Output::getValues() const
{
	return values;
}
