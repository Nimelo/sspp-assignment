#include "InPlaceSorterTest.h"
#include "Definitions.h"
#include <gtest\gtest.h>

TEST_F(InPlaceSorterTest, shouldSortProperly) {
  const int N = 5;
  INDEXING_TYPE I[N] = { 5, 4, 3, 2, 1 }, J[N] = { 1, 2, 3, 4, 5 };
  INDEXING_TYPE correctI[N] = { 1, 2, 3, 4, 5 }, correctJ[N] = { 5, 4, 3, 2, 1 };
  FLOATING_TYPE V[N] = { 1, 2, 3, 4, 5 }, correctV[N] = { 5, 4, 3, 2, 1 };

  sorter->Sort(I, J, V, N);

  assertArrays(correctI, I, N, "I -> Incorrect value at: ");
  assertArrays(correctJ, J, N, "J -> Incorrect value at: ");
  assertArrays(correctV, V, N, "V -> Incorrect value tach: ");

}

TEST_F(InPlaceSorterTest, shouldSortProperly2) {
  const int N = 5;
  INDEXING_TYPE I[N] = { 3, 2, 2, 1, 1 }, J[N] = { 5, 2, 1, 9, 9 };
  INDEXING_TYPE correctI[N] = { 1, 1, 2, 2, 3 }, correctJ[N] = { 9, 9, 1, 2, 5 };
  FLOATING_TYPE V[N] = { 1, 2, 3, 4, 5 }, correctV[N] = { 4, 5, 3, 2, 1 };

  sorter->Sort(I, J, V, N);

  assertArrays(correctI, I, N, "I -> Incorrect value at: ");
  assertArrays(correctJ, J, N, "J -> Incorrect value at: ");
  assertArrays(correctV, V, N, "V -> Incorrect value tach: ");

}