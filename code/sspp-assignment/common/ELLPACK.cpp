#include "ELLPACK.h"

void representations::ellpack::ELLPACK::rewrite(ELLPACK & lhs, const ELLPACK & rhs)
{
	lhs.M = rhs.M;
	lhs.N = rhs.N;
	lhs.NZ = rhs.NZ;
	lhs.MAXNZ = rhs.MAXNZ;

	lhs.AS = new FLOATING_TYPE*[rhs.M];
	for (int i = 0; i < rhs.M; i++)
	{
		lhs.AS[i] = new FLOATING_TYPE[rhs.MAXNZ];
		for (int j = 0; j < rhs.MAXNZ; j++)
			lhs.AS[i][j] = rhs.AS[i][j];
	}

	lhs.JA = new int*[rhs.M];
	for (int i = 0; i < rhs.M; i++)
	{
		lhs.JA[i] = new int[rhs.MAXNZ];
		for (int j = 0; j < rhs.MAXNZ; j++)
			lhs.JA[i][j] = rhs.JA[i][j];
	}
}

representations::ellpack::ELLPACK::ELLPACK(int M, int N, int NZ, int MAXNZ, int ** JA, FLOATING_TYPE ** AS)
	: M(M), N(N), NZ(NZ), MAXNZ(MAXNZ), JA(JA), AS(AS)
{
}

representations::ellpack::ELLPACK::ELLPACK(const ELLPACK & other)
{
	rewrite(*this, other);
}

representations::ellpack::ELLPACK & representations::ellpack::ELLPACK::operator=(representations::ellpack::ELLPACK rhs)
{
	rewrite(*this, rhs);
	return *this;
}

representations::ellpack::ELLPACK::~ELLPACK()
{
	for (int i = 0; i < M; i++)
		delete[] JA[i];
	delete[] JA;

	for (int i = 0; i < M; i++)
		delete[] AS[i];
	delete[] AS;
}
