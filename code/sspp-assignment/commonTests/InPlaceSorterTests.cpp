#include "InPlaceSorterTest.h"
#include "Definitions.h"
#include <gtest\gtest.h>

TEST_F(InPlaceSorterTest, shouldSortProperly)
{
	const int N = 5;
	int I[N] = { 5, 4, 3, 2, 1 }, J[N] = { 1, 2, 3, 4, 5 };
	int correctI[N] = { 1, 2, 3, 4, 5 }, correctJ[N] = { 5, 4, 3, 2, 1 };
	FLOATING_TYPE V[N] = { 1, 2, 3, 4, 5 }, correctV[N] = { 5, 4, 3, 2, 1 };
	

	sorter->sort(I, J, V, N);

	assertArrays(correctI, I, N, "I -> Incorrect value at: ");
	assertArrays(correctJ, J, N, "J -> Incorrect value at: ");
	assertArrays(correctV, V, N, "V -> Incorrect value tach: ");

}