#include "IntermediarySparseMatrix.h"

representations::intermediary::IntermediarySparseMatrix::IntermediarySparseMatrix()
{
}

representations::intermediary::IntermediarySparseMatrix::IntermediarySparseMatrix(int m, int n, int nz, int * iIndexes, int * jIndexes, FLOATING_TYPE * values)
	: M(m), N(n), NZ(nz), IIndexes(iIndexes), JIndexes(jIndexes), Values(values)
{
}

representations::intermediary::IntermediarySparseMatrix::IntermediarySparseMatrix(const IntermediarySparseMatrix & other)
{
	this->N = other.N;
	this->M = other.M;
	this->NZ = other.NZ;

	this->IIndexes = new int[NZ];
	this->JIndexes = new int[NZ];
	this->Values = new FLOATING_TYPE[NZ];

	for (int it = 0; it < NZ; it++)
	{
		IIndexes[it] = other.IIndexes[it];
		JIndexes[it] = other.JIndexes[it];
		Values[it] = other.Values[it];
	}
}

representations::intermediary::IntermediarySparseMatrix & representations::intermediary::IntermediarySparseMatrix::operator=(representations::intermediary::IntermediarySparseMatrix rhs)
{
	this->N = rhs.N;
	this->M = rhs.M;
	this->NZ = rhs.NZ;

	this->IIndexes = new int[NZ];
	this->JIndexes = new int[NZ];
	this->Values = new FLOATING_TYPE[NZ];

	for (int it = 0; it < NZ; it++)
	{
		IIndexes[it] = rhs.IIndexes[it];
		JIndexes[it] = rhs.JIndexes[it];
		Values[it] = rhs.Values[it];
	}

	return *this;
}

representations::intermediary::IntermediarySparseMatrix::~IntermediarySparseMatrix()
{
	if(IIndexes != 0)
		delete[] IIndexes;
	if(JIndexes != 0)
		delete[] JIndexes;
	if(Values != 0)
		delete[] Values;
}