#include "InPlaceSorterTest.h"
#include "Definitions.h"
#include <gtest\gtest.h>

TEST_F(InPlaceSorterTest, shouldSortProperly) {
  const int N = 5;
  std::vector<INDEXING_TYPE> I = { 5, 4, 3, 2, 1 }, J = { 1, 2, 3, 4, 5 };
  std::vector<INDEXING_TYPE> correctI = { 1, 2, 3, 4, 5 }, correctJ = { 5, 4, 3, 2, 1 };
  std::vector<FLOATING_TYPE> V = { 1, 2, 3, 4, 5 }, correctV = { 5, 4, 3, 2, 1 };

  sorter->Sort(I, J, V, N);

  assertArrays(&correctI[0], &I[0], N, "I -> Incorrect value at: ");
  assertArrays(&correctJ[0], &J[0], N, "J -> Incorrect value at: ");
  assertArrays(&correctV[0], &V[0], N, "V -> Incorrect value tach: ");
}

TEST_F(InPlaceSorterTest, shouldSortProperly2) {
  const int N = 5;
  std::vector<INDEXING_TYPE> I = { 3, 2, 2, 1, 1 }, J = { 5, 2, 1, 9, 9 };
  std::vector<INDEXING_TYPE> correctI = { 1, 1, 2, 2, 3 }, correctJ = { 9, 9, 1, 2, 5 };
  std::vector<FLOATING_TYPE> V = { 1, 2, 3, 4, 5 }, correctV = { 4, 5, 3, 2, 1 };

  sorter->Sort(I, J, V, N);

  assertArrays(&correctI[0], &I[0], N, "I -> Incorrect value at: ");
  assertArrays(&correctJ[0], &J[0], N, "J -> Incorrect value at: ");
  assertArrays(&correctV[0], &V[0], N, "V -> Incorrect value tach: ");
}