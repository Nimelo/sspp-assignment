#include "Output.h"

representations::output::Output::Output()
{
}

representations::output::Output::Output(int size, FLOATING_TYPE * values)
	: Values(values), N(size)
{
}

representations::output::Output::Output(const Output & other)
{
	N = other.N;
	Values = new FLOATING_TYPE[N];
	for (auto i = 0; i < N; i++)
		Values[i] = other.Values[i];
}

representations::output::Output & representations::output::Output::operator=(representations::output::Output other)
{
	N = other.N;
	Values = new FLOATING_TYPE[N];
	for (auto i = 0; i < N; i++)
		Values[i] = other.Values[i];

	return *this;
}

representations::output::Output::~Output()
{
	if(Values != 0)
		delete[] Values;
}

std::ostream & representations::output::operator<<(std::ostream & os, const Output & o)
{
	for (auto i = 0; i < o.N; i++)
		os << o.Values[i] << SPACE;

	return os;
}
