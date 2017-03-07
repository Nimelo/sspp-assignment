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

representations::ellpack::ELLPACK::ELLPACK()
	: M(0), N(0), NZ(0), MAXNZ(0), JA(0), AS(0)
{

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

std::ostream & representations::ellpack::operator<<(std::ostream & os, const ELLPACK & ellpack)
{
	os << ellpack.M << LINE_SEPARATOR;
	os << ellpack.N << LINE_SEPARATOR;
	os << ellpack.NZ << LINE_SEPARATOR;
	os << ellpack.MAXNZ << LINE_SEPARATOR;

	for (int i = 0; i < ellpack.M; i++)
	{
		for (int j = 0; j < ellpack.MAXNZ - 1; j++)
		{
			os << ellpack.JA[i][j] << SPACE;
		}
		os << ellpack.JA[i][ellpack.MAXNZ - 1] << LINE_SEPARATOR;
	}

	for (int i = 0; i < ellpack.M; i++)
	{
		for (int j = 0; j < ellpack.MAXNZ - 1; j++)
		{
			os << ellpack.AS[i][j] << SPACE;
		}
		os << ellpack.AS[i][ellpack.MAXNZ - 1] << LINE_SEPARATOR;
	}

	return os;
}

std::istream & representations::ellpack::operator >> (std::istream & is, ELLPACK & ellpack)
{
	int **JA;
	FLOATING_TYPE **AS;

	is >> ellpack.M;
	is >> ellpack.N;
	is >> ellpack.NZ;
	is >> ellpack.MAXNZ;

	JA = new int*[ellpack.M];
	for (int i = 0; i < ellpack.M; i++)
		JA[i] = new int[ellpack.MAXNZ];

	AS = new FLOATING_TYPE*[ellpack.M];
	for (int i = 0; i < ellpack.M; i++)
		AS[i] = new FLOATING_TYPE[ellpack.MAXNZ];

	for (int i = 0; i < ellpack.M; i++)
	{
		for (int j = 0; j < ellpack.MAXNZ; j++)
		{
			is >> JA[i][j];
		}
	}

	for (int i = 0; i < ellpack.M; i++)
	{
		for (int j = 0; j < ellpack.MAXNZ; j++)
		{
			is >> AS[i][j];
		}
	}

	if (ellpack.JA != 0)
	{
		for (int i = 0; i < ellpack.M; i++)
			delete[] ellpack.JA[i];
		delete[] ellpack.JA;
	}

	if (ellpack.AS != 0)
	{
		for (int i = 0; i < ellpack.M; i++)
			delete[] ellpack.AS[i];
		delete[] ellpack.AS;
	}

	ellpack.JA = JA;
	ellpack.AS = AS;

	return is;
}
