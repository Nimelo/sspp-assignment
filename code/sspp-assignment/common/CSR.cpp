#include "CSR.h"

representations::csr::CSR::CSR(int NZ, int M, int N, int * IRP, int * JA, FLOATING_TYPE * AS)
	: NZ(NZ), M(M), N(N), IRP(IRP), JA(JA), AS(AS)
{
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
