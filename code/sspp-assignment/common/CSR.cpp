#include "CSR.h"

representations::csr::CSR::CSR(int NZ, int M, int N, int * IRP, int * JA, FLOATING_TYPE * AS)
	: NZ(NZ), M(M), N(N), IRP(IRP), JA(JA), AS(AS)
{
}

representations::csr::CSR::CSR(const CSR & other)
{
	this->NZ = other.getASSize();
	this->M = other.getM();
	this->N = other.getN();
	this->IRP = new int[other.getIRPSize()];
	for (int i = 0; i < other.getIRPSize(); i++)
		this->IRP[i] = other.getIRP()[i];

	this->JA = new int[other.getJASize()];
	for (int i = 0; i < other.getJASize(); i++)
		this->JA[i] = other.getJA()[i];

	this->AS = new FLOATING_TYPE[other.getASSize()];
	for (int i = 0; i < other.getASSize(); i++)
		this->AS[i] = other.getAS()[i];
}

representations::csr::CSR & representations::csr::CSR::operator=(CSR rhs)
{
	// TODO: insert return statement here
	return *this;
}

representations::csr::CSR::~CSR()
{
	delete[] IRP;
	delete[] JA;
	delete[] AS;
}

int representations::csr::CSR::getM() const
{
	return M;
}

int representations::csr::CSR::getN() const
{
	return N;
}

int * representations::csr::CSR::getIRP() const
{
	return IRP;
}

int * representations::csr::CSR::getJA() const
{
	return JA;
}

FLOATING_TYPE * representations::csr::CSR::getAS() const
{
	return AS;
}

int representations::csr::CSR::getIRPSize() const
{
	return M + 1;
}

int representations::csr::CSR::getJASize() const
{
	return NZ;
}

int representations::csr::CSR::getASSize() const
{
	return NZ;
}
