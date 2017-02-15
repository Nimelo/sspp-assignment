#include "CSR.h"

void representations::csr::CSR::rewrite(CSR & lhs, const CSR & rhs)
{
	lhs.NZ = rhs.getASSize();
	lhs.M = rhs.M;
	lhs.N = rhs.N;
	lhs.IRP = new int[rhs.getIRPSize()];
	for (int i = 0; i < rhs.getIRPSize(); i++)
		lhs.IRP[i] = rhs.IRP[i];

	lhs.JA = new int[rhs.getJASize()];
	for (int i = 0; i < rhs.getJASize(); i++)
		lhs.JA[i] = rhs.JA[i];

	lhs.AS = new FLOATING_TYPE[rhs.getASSize()];
	for (int i = 0; i < rhs.getASSize(); i++)
		lhs.AS[i] = rhs.AS[i];
}

representations::csr::CSR::CSR(int NZ, int M, int N, int * IRP, int * JA, FLOATING_TYPE * AS)
	: NZ(NZ), M(M), N(N), IRP(IRP), JA(JA), AS(AS)
{
}

representations::csr::CSR::CSR(const CSR & other)
{
	rewrite(*this, other);
}

representations::csr::CSR & representations::csr::CSR::operator=(CSR rhs)
{
	rewrite(*this, rhs);
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
