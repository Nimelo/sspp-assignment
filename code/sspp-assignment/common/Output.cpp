#include "Output.h"

representations::output::Output::Output(int size, FLOATING_TYPE * values)
	: N(size), Values(values)
{
}

representations::output::Output::Output(const Output & other)
{
	N = other.N;
	Values = new FLOATING_TYPE[N];
	for (int i = 0; i < N; i++)
		Values[i] = other.Values[i];
}

representations::output::Output::~Output()
{
	if(Values != 0)
		delete[] Values;
}

std::ostream & representations::output::operator<<(std::ostream & os, const Output & o)
{
	for (int i = 0; i < o.N; i++)
	{
		os << o.Values[i] << std::endl;
	}
	
	return os;
}
