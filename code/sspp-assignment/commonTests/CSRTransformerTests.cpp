#include "CSRTransformerTest.h"
#include "Definitions.h"
#include <gtest\gtest.h>

TEST_F(CSRTransformerTest, shouldTransformCorrectly_Salvatore)
{
	const int M = 4, N = 4, NZ = 7;
	int *iIndexes = new int[NZ], *jIndexes = new int[NZ];
	FLOATING_TYPE *values = new FLOATING_TYPE[NZ];
	iIndexes[0] = 0; iIndexes[1] = 0; iIndexes[2] = 1; iIndexes[3] = 1; iIndexes[4] = 2; iIndexes[5] = 3; iIndexes[6] = 3;
	jIndexes[0] = 0; jIndexes[1] = 1; jIndexes[2] = 1; jIndexes[3] = 2; jIndexes[4] = 2; jIndexes[5] = 2; jIndexes[6] = 3;
	values[0] = 11; values[1] = 12; values[2] = 22; values[3] = 23; values[4] = 33; values[5] = 43; values[6] = 44;
	int correctIRP[5] = { 0, 2, 4, 5, 8 };
	int correctJA[NZ] = { 0, 1, 1, 2, 2, 2, 3 };

	representations::intermediary::IntermediarySparseMatrix ism(M, N, NZ, iIndexes, jIndexes, values);
	auto csr = csrTransformer->transform(ism);

	ASSERT_EQ(M, csr.getM()) << "M values is different.";
	ASSERT_EQ(N, csr.getN()) << "N values is different.";
	ASSERT_EQ(NZ, csr.getJASize()) << "JASize values is different.";
	ASSERT_EQ(NZ, csr.getASSize()) << "ASSize values is different.";
	ASSERT_EQ(M + 1, csr.getIRPSize()) << "IRPSize values is different.";

	assertArrays(values, csr.getAS(), NZ, "AS -> Incorrect value at: ");
	assertArrays(correctIRP, csr.getIRP(), 5, "IRP -> Incorrect value at: ");
	assertArrays(correctJA, csr.getJA(), NZ, "JA -> Incorrect value at: ");
}

TEST_F(CSRTransformerTest, shouldTransformCorrectly)
{
	const int M = 3, N = 4, NZ = 5;
	int *iIndexes = new int[NZ], *jIndexes = new int[NZ];
	FLOATING_TYPE *values = new FLOATING_TYPE[NZ];
	iIndexes[0] = 0; iIndexes[1] = 1; iIndexes[2] = 1; iIndexes[3] = 2; iIndexes[4] = 2;
	jIndexes[0] = 2; jIndexes[1] = 2; jIndexes[2] = 3; jIndexes[3] = 0; jIndexes[4] = 1;
	values[0] = 1; values[1] = 2; values[2] = 3; values[3] = 4; values[4] = 1;
	int correctIRP[4] = { 0, 1, 3, 6 };
	int correctJA[NZ] = {2, 2, 3, 0, 1};

	representations::intermediary::IntermediarySparseMatrix ism(M, N, NZ, iIndexes, jIndexes, values);
	auto csr = csrTransformer->transform(ism);
	
	ASSERT_EQ(M, csr.getM()) << "M values is different.";
	ASSERT_EQ(N, csr.getN()) << "N values is different.";
	ASSERT_EQ(NZ, csr.getJASize()) << "JASize values is different.";
	ASSERT_EQ(NZ, csr.getASSize()) << "ASSize values is different.";
	ASSERT_EQ(M+1, csr.getIRPSize()) << "IRPSize values is different.";

	assertArrays(values, csr.getAS(), NZ, "AS -> Incorrect value at: ");
	assertArrays(correctIRP, csr.getIRP(), 4, "IRP -> Incorrect value at: ");
	assertArrays(correctJA, csr.getJA(), NZ, "JA -> Incorrect value at: ");
}