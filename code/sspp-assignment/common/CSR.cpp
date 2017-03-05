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

representations::csr::CSR::CSR()
{
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

std::ostream & representations::csr::operator<<(std::ostream & os, const CSR & csr)
{	
	os << csr.M << LINE_SEPARATOR;
	os << csr.N << LINE_SEPARATOR;
	os << csr.NZ << LINE_SEPARATOR;

	for (int i = 0; i < csr.M; i++)
		os << csr.IRP[i] << SPACE;
	os << csr.IRP[csr.M] << LINE_SEPARATOR;

	for (int i = 0; i < csr.NZ - 1; i++)
		os << csr.JA[i] << SPACE;
	os << csr.JA[csr.NZ-1] << LINE_SEPARATOR;

	for (int i = 0; i < csr.NZ - 1; i++)
		os << csr.AS[i] << SPACE;
	os << csr.AS[csr.NZ - 1] << LINE_SEPARATOR;

	return os;
}

std::istream & representations::csr::operator >> (std::istream & is, CSR & csr)
{
	int *IRP, *JA;
	FLOATING_TYPE *AS;

	is >> csr.M;
	is >> csr.N;
	is >> csr.NZ;

	IRP = new int[csr.M + 1];
	JA = new int[csr.NZ];
	AS = new FLOATING_TYPE[csr.NZ];

	for (int i = 0; i < csr.M + 1; i++)
		is >> IRP[i];
	for (int i = 0; i < csr.NZ; i++)
		is >> JA[i];
	for (int i = 0; i < csr.NZ; i++)
		is >> AS[i];

	if (csr.AS != 0)
		delete[] csr.AS;
	if (csr.IRP != 0)
		delete[] csr.IRP;
	if (csr.JA != 0)
		delete[] csr.JA;

	csr.AS = AS;
	csr.IRP = IRP;
	csr.JA = JA;

	return is;
}
