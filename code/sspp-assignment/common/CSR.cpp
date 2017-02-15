#include "CSR.h"

representations::csr::CSR::CSR(int NZ, int M, int N, int * IRP, int * JA, FLOATING_TYPE * AS)
	: NZ(NZ), M(M), N(N), IRP(IRP), JA(JA), AS(AS)
{
}

representations::csr::CSR::CSR(const CSR & other)
{
	this->NZ = other.getASSize();
	this->M = other.M;
	this->N = other.N;
	this->IRP = new int[other.getIRPSize()];
	for (int i = 0; i < other.getIRPSize(); i++)
		this->IRP[i] = other.IRP[i];

	this->JA = new int[other.getJASize()];
	for (int i = 0; i < other.getJASize(); i++)
		this->JA[i] = other.JA[i];

	this->AS = new FLOATING_TYPE[other.getASSize()];
	for (int i = 0; i < other.getASSize(); i++)
		this->AS[i] = other.AS[i];
}

representations::csr::CSR & representations::csr::CSR::operator=(CSR rhs)
{
	this->NZ = rhs.getASSize();
	this->M = rhs.M;
	this->N = rhs.N;
	this->IRP = new int[rhs.getIRPSize()];
	for (int i = 0; i < rhs.getIRPSize(); i++)
		this->IRP[i] = rhs.IRP[i];

	this->JA = new int[rhs.getJASize()];
	for (int i = 0; i < rhs.getJASize(); i++)
		this->JA[i] = rhs.JA[i];

	this->AS = new FLOATING_TYPE[rhs.getASSize()];
	for (int i = 0; i < rhs.getASSize(); i++)
		this->AS[i] = rhs.AS[i];
	return *this;
}

representations::csr::CSR::~CSR()
{
	delete[] IRP;
	delete[] JA;
	delete[] AS;
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
