#include "MatrixMarketReaderTest.h"
#include <fstream>
#include "..\common\mmio.h"
#include "..\common\Definitions.h"

TEST_F(MatrixMarketReaderTest, CRG)
{
	const char * fileName = "matrixMarketTestCRG";
	const int M = 5, N = 5, NZ = 2;
	FLOATING_TYPE VAL[] = { 0, 1 };
	int I[] = { 5, 4 };
	int J[] = { 3, 2 };

	std::fstream fs;
	fs.open(fileName, std::fstream::out | std::fstream::trunc);
	fs << MatrixMarketBanner << SPACE
		<< MM_MTX_STR << SPACE
		<< MM_COORDINATE_STR << SPACE
		<< MM_REAL_STR << SPACE
		<< MM_GENERAL_STR << std::endl;
	fs << M << SPACE << N << SPACE << NZ << std::endl;
	for (int i = 0; i < NZ; i++)
	{
		fs << I[i] << SPACE << J[i] << SPACE << VAL[i] << std::endl;
	}
	fs.close();

	try {
		auto ism = this->matrixMarketReader->fromFile(fileName);

		ASSERT_EQ(M, ism.M);
		ASSERT_EQ(N, ism.N);
		ASSERT_EQ(NZ, ism.NZ);
		assertArrays(I, ism.IIndexes, NZ, "");
		assertArrays(J, ism.JIndexes, NZ, "");
		assertArrays(VAL, ism.Values, NZ, "");
	}
	catch (...)
	{
		std::remove(fileName);
		FAIL();
	}
	std::remove(fileName);
}

TEST_F(MatrixMarketReaderTest, CRS)
{
	const char * fileName = "matrixMarketTestCRS";
	const int M = 5, N = 5, NZ = 2;
	FLOATING_TYPE VAL[] = { 0, 1 };
	int I[] = { 5, 4 };
	int J[] = { 3, 2 };

	std::fstream fs;
	fs.open(fileName, std::fstream::out | std::fstream::trunc);
	fs << MatrixMarketBanner << SPACE
		<< MM_MTX_STR << SPACE
		<< MM_COORDINATE_STR << SPACE
		<< MM_REAL_STR << SPACE
		<< MM_SYMM_STR << std::endl;
	fs << M << SPACE << N << SPACE << NZ << std::endl;
	for (int i = 0; i < NZ; i++)
	{
		fs << I[i] << SPACE << J[i] << SPACE << VAL[i] << std::endl;
	}
	fs.close();

	try {
		auto ism = this->matrixMarketReader->fromFile(fileName);

		ASSERT_EQ(M, ism.M);
		ASSERT_EQ(N, ism.N);
		ASSERT_EQ(NZ << 1, ism.NZ);
		assertArrays(I, ism.IIndexes, NZ, "");
		assertArrays(J, ism.JIndexes, NZ, "");
		assertArrays(VAL, ism.Values, NZ, "");

		assertArrays(I, ism.JIndexes + NZ, NZ, "");
		assertArrays(J, ism.IIndexes + NZ, NZ, "");
		assertArrays(VAL, ism.Values + NZ, NZ, "");
	}
	catch (...)
	{
		std::remove(fileName);
		FAIL();
	}
	std::remove(fileName);
}


TEST_F(MatrixMarketReaderTest, CPS)
{
	const char * fileName = "matrixMarketTestCPS";
	const int M = 5, N = 5, NZ = 2;
	FLOATING_TYPE VAL[] = { 0, 1 };
	int I[] = { 5, 4 };
	int J[] = { 3, 2 };

	std::fstream fs;
	fs.open(fileName, std::fstream::out | std::fstream::trunc);
	fs << MatrixMarketBanner << SPACE
		<< MM_MTX_STR << SPACE
		<< MM_COORDINATE_STR << SPACE
		<< MM_PATTERN_STR << SPACE
		<< MM_SYMM_STR << std::endl;
	fs << M << SPACE << N << SPACE << NZ << std::endl;
	for (int i = 0; i < NZ; i++)
	{
		fs << I[i] << SPACE << J[i] << SPACE /*<< VAL[i]*/ << std::endl;
	}
	fs.close();

	try {
		auto ism = this->matrixMarketReader->fromFile(fileName);

		ASSERT_EQ(M, ism.M);
		ASSERT_EQ(N, ism.N);
		ASSERT_EQ(NZ << 1, ism.NZ);
		assertArrays(I, ism.IIndexes, NZ, "");
		assertArrays(J, ism.JIndexes, NZ, "");

		assertArrays(I, ism.JIndexes + NZ, NZ, "");
		assertArrays(J, ism.IIndexes + NZ, NZ, "");
		assertArrays(ism.Values, ism.Values + NZ, NZ, "");
	}
	catch (...)
	{
		std::remove(fileName);
		FAIL();
	}
	std::remove(fileName);
}

TEST_F(MatrixMarketReaderTest, CPG)
{
	const char * fileName = "matrixMarketTestCPG";
	const int M = 5, N = 5, NZ = 2;
	FLOATING_TYPE VAL[] = { 0, 1 };
	int I[] = { 5, 4 };
	int J[] = { 3, 2 };

	std::fstream fs;
	fs.open(fileName, std::fstream::out | std::fstream::trunc);
	fs << MatrixMarketBanner << SPACE
		<< MM_MTX_STR << SPACE
		<< MM_COORDINATE_STR << SPACE
		<< MM_PATTERN_STR << SPACE
		<< MM_GENERAL_STR << std::endl;
	fs << M << SPACE << N << SPACE << NZ << std::endl;
	for (int i = 0; i < NZ; i++)
	{
		fs << I[i] << SPACE << J[i] << SPACE /*<< VAL[i]*/ << std::endl;
	}
	fs.close();

	try {
		auto ism = this->matrixMarketReader->fromFile(fileName);

		ASSERT_EQ(M, ism.M);
		ASSERT_EQ(N, ism.N);
		ASSERT_EQ(NZ, ism.NZ);
		assertArrays(I, ism.IIndexes, NZ, "");
		assertArrays(J, ism.JIndexes, NZ, "");
	}
	catch (...)
	{
		std::remove(fileName);
		FAIL();
	}
	std::remove(fileName);
}