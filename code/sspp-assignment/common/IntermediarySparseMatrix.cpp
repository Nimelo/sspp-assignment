#include "IntermediarySparseMatrix.h"

representations::intermediary::IntermediarySparseMatrix::IntermediarySparseMatrix(int m, int n, int nz, int * iIndexes, int * jIndexes, FLOATING_TYPE * values)
	: m(m), n(n), nz(nz), i(iIndexes), j(jIndexes), values(values)
{
}

representations::intermediary::IntermediarySparseMatrix::IntermediarySparseMatrix(const IntermediarySparseMatrix & other)
{
	this->n = other.n;
	this->m = other.m;
	this->nz = other.nz;

	this->i = new int[nz];
	this->j = new int[nz];
	this->values = new FLOATING_TYPE[nz];

	for (int it = 0; it < nz; it++)
	{
		i[it] = other.getIIndexes()[it];
		j[it] = other.getJIndexes()[it];
		values[it] = other.getValues()[it];
	}
}

representations::intermediary::IntermediarySparseMatrix::~IntermediarySparseMatrix()
{
	delete[] i;
	delete[] j;
	delete[] values;
}

int representations::intermediary::IntermediarySparseMatrix::getNZ() const
{
	return nz;
}

int representations::intermediary::IntermediarySparseMatrix::getM() const
{
	return m;
}

int representations::intermediary::IntermediarySparseMatrix::getN() const
{
	return n;
}

int * representations::intermediary::IntermediarySparseMatrix::getIIndexes() const
{
	return i;
}

int * representations::intermediary::IntermediarySparseMatrix::getJIndexes() const
{
	return j;
}

FLOATING_TYPE * representations::intermediary::IntermediarySparseMatrix::getValues() const
{
	return values;
}
