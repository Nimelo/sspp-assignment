#include "InPlaceSorterTest.h"
#include "Definitions.h"
#include <gtest\gtest.h>
#include "DebugTools.h"

TEST_F(InPlaceSorterTest, shouldSortProperly)
{
	const int N = 5;
	int I[N] = { 5, 4, 3, 2, 1 }, J[N] = { 1, 2, 3, 4, 5 };
	int correctI[N] = { 1, 2, 3, 4, 5 }, correctJ[N] = { 5, 4, 3, 2, 1 };
	FLOATING_TYPE V[N] = { 1, 2, 3, 4, 5 }, correctV[N] = { 5, 4, 3, 2, 1 };
	

	sorter->sort(I, J, V, N);
	tools::debug::DebugTools debug;

	debug.printArray(I, N, "I");
	debug.printArray(J, N, "J");
	debug.printArray(V, N, "V");

	debug.printArray(correctI, N, "correctI");
	debug.printArray(correctJ, N, "correctJ");
	debug.printArray(correctV, N, "correctV");

	assertArrays(correctI, I, N, "I -> Incorrect value at: ");
	assertArrays(correctJ, J, N, "J -> Incorrect value at: ");
	assertArrays(correctV, V, N, "V -> Incorrect value tach: ");

}

TEST_F(InPlaceSorterTest, shouldSortProperly2)
{
	const int N = 5;
	int I[N] = { 3, 2, 2, 1, 1 }, J[N] = { 5, 2, 1, 9, 9 };
	int correctI[N] = { 1, 1, 2, 2, 3 }, correctJ[N] = { 9, 9, 1, 2, 5 };
	FLOATING_TYPE V[N] = { 1, 2, 3, 4, 5 }, correctV[N] = { 4, 5, 3, 2, 1 };


	sorter->sort(I, J, V, N);
	tools::debug::DebugTools debug;

	debug.printArray(I, N, "I");
	debug.printArray(J, N, "J");
	debug.printArray(V, N, "V");

	debug.printArray(correctI, N, "correctI");
	debug.printArray(correctJ, N, "correctJ");
	debug.printArray(correctV, N, "correctV");

	assertArrays(correctI, I, N, "I -> Incorrect value at: ");
	assertArrays(correctJ, J, N, "J -> Incorrect value at: ");
	assertArrays(correctV, V, N, "V -> Incorrect value tach: ");

}