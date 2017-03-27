#include "InPlaceSorterTest.h"

#include <gtest\gtest.h>
#include <gmock/gmock.h>

TEST_F(InPlaceSorterTest, shouldSortProperly) {
  const int N = 5;
  std::vector<unsigned> I = { 5, 4, 3, 2, 1 }, J = { 1, 2, 3, 4, 5 };
  std::vector<unsigned> correctI = { 1, 2, 3, 4, 5 }, correctJ = { 5, 4, 3, 2, 1 };
  std::vector<float> V = { 1, 2, 3, 4, 5 }, correctV = { 5, 4, 3, 2, 1 };

  sorter->InsertionSort<float>(I, J, V);

  ASSERT_THAT(correctI, ::testing::ContainerEq(I));
  ASSERT_THAT(correctJ, ::testing::ContainerEq(J));
  ASSERT_THAT(correctV, ::testing::ContainerEq(V));
}

TEST_F(InPlaceSorterTest, shouldSortProperly2) {
  const int N = 5;
  std::vector<unsigned> I = { 3, 2, 2, 1, 1 }, J = { 5, 2, 1, 9, 9 };
  std::vector<unsigned> correctI = { 1, 1, 2, 2, 3 }, correctJ = { 9, 9, 2, 1, 5 };
  std::vector<float> V = { 1, 2, 3, 4, 5 }, correctV = { 4, 5, 2, 3, 1 };

  sorter->InsertionSort<float>(I, J, V);

  ASSERT_THAT(correctI, ::testing::ContainerEq(I));
  ASSERT_THAT(correctJ, ::testing::ContainerEq(J));
  ASSERT_THAT(correctV, ::testing::ContainerEq(V));
}