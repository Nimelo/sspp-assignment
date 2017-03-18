#include "MatrixMarketTest.h"
#include "MatrixMarket.h"
#include "../common/MatrixMarketHeader.h"
#include "../common/MatrixMarket.h"

#include "..\common\Definitions.h"
#include <sstream>
using namespace sspp::io;

TEST_F(MatrixMarketTest, CRG) {
  INDEXING_TYPE c_rows = 5, c_columns = 5, c_non_zeros = 2,
    rows, columns, non_zeros;
  std::vector<FLOATING_TYPE> c_values = { 0, 1 },
    values;
  std::vector<INDEXING_TYPE> c_row_indicies = { 5, 4 },
    c_column_indicies = { 3, 2 },
    row_indicies,
    column_indicies;
  MatrixMarketHeader mmh(Matrix, Sparse, Real, General);
  WriteHeader(mmh);
  WriteIndices(c_rows, c_columns, c_non_zeros, c_row_indicies, c_column_indicies, c_values);
  MatrixMarket mm(mmh);

  mm.ReadIndices(ss, rows, columns, non_zeros, 
                 row_indicies, column_indicies, values);

  ASSERT_EQ(c_rows, rows);
  ASSERT_EQ(c_columns, columns);
  ASSERT_EQ(c_non_zeros, non_zeros);
  ASSERT_EQ(c_non_zeros, row_indicies.size());
  ASSERT_EQ(c_non_zeros, column_indicies.size());
  ASSERT_EQ(c_non_zeros, values.size());
  assertArrays(&c_row_indicies[0], &row_indicies[0], c_non_zeros, "");
  assertArrays(&c_column_indicies[0], &column_indicies[0], c_non_zeros, "");
  assertArrays(&c_values[0], &values[0], c_non_zeros, "");
}

TEST_F(MatrixMarketTest, CRS) {
  INDEXING_TYPE c_rows = 5, c_columns = 5, c_non_zeros = 2,
    rows, columns, non_zeros;
  std::vector<FLOATING_TYPE> c_values = { 0, 1 },
    values;
  std::vector<INDEXING_TYPE> c_row_indicies = { 5, 4 },
    c_column_indicies = { 3, 2 },
    row_indicies,
    column_indicies;
  MatrixMarketHeader mmh(Matrix, Sparse, Real, Symetric);
  WriteHeader(mmh);
  WriteIndices(c_rows, c_columns, c_non_zeros, c_row_indicies, c_column_indicies, c_values);
  MatrixMarket mm(mmh);

  mm.ReadIndices(ss, rows, columns, non_zeros,
                 row_indicies, column_indicies, values);

  ASSERT_EQ(c_rows, rows);
  ASSERT_EQ(c_columns, columns);
  ASSERT_EQ(c_non_zeros << 1, non_zeros);
  ASSERT_EQ(c_non_zeros << 1, row_indicies.size());
  ASSERT_EQ(c_non_zeros << 1, column_indicies.size());
  ASSERT_EQ(c_non_zeros << 1, values.size());
  assertArrays(&c_row_indicies[0], &row_indicies[0], c_non_zeros, "");
  assertArrays(&c_column_indicies[0], &row_indicies[0] + c_non_zeros, c_non_zeros, "");
  assertArrays(&c_column_indicies[0], &column_indicies[0], c_non_zeros, "");
  assertArrays(&c_row_indicies[0], &column_indicies[0] + c_non_zeros, c_non_zeros, "");
  assertArrays(&c_values[0], &values[0], c_non_zeros, "");
  assertArrays(&c_values[0], &values[0] + c_non_zeros, c_non_zeros, "");
}


TEST_F(MatrixMarketTest, CPS) {
  INDEXING_TYPE c_rows = 5, c_columns = 5, c_non_zeros = 2,
    rows, columns, non_zeros;
  std::vector<FLOATING_TYPE> c_values = { 0, 1 },
    values;
  std::vector<INDEXING_TYPE> c_row_indicies = { 5, 4 },
    c_column_indicies = { 3, 2 },
    row_indicies,
    column_indicies;
  MatrixMarketHeader mmh(Matrix, Sparse, Pattern, Symetric);
  WriteHeader(mmh);
  WriteIndicesPattern(c_rows, c_columns, c_non_zeros, c_row_indicies, c_column_indicies);
  MatrixMarket mm(mmh);

  mm.ReadIndices(ss, rows, columns, non_zeros,
                 row_indicies, column_indicies, values);

  ASSERT_EQ(c_rows, rows);
  ASSERT_EQ(c_columns, columns);
  ASSERT_EQ(c_non_zeros << 1, non_zeros);
  ASSERT_EQ(c_non_zeros << 1, row_indicies.size());
  ASSERT_EQ(c_non_zeros << 1, column_indicies.size());
  ASSERT_EQ(c_non_zeros << 1, values.size());
  assertArrays(&c_row_indicies[0], &row_indicies[0], c_non_zeros, "");
  assertArrays(&c_column_indicies[0], &row_indicies[0] + c_non_zeros, c_non_zeros, "");
  assertArrays(&c_column_indicies[0], &column_indicies[0], c_non_zeros, "");
  assertArrays(&c_row_indicies[0], &column_indicies[0] + c_non_zeros, c_non_zeros, "");
  assertArrays(&values[0], &values[0], c_non_zeros, "");
  assertArrays(&values[0], &values[0] + c_non_zeros, c_non_zeros, "");
}

TEST_F(MatrixMarketTest, CPG) {
  INDEXING_TYPE c_rows = 5, c_columns = 5, c_non_zeros = 2,
    rows, columns, non_zeros;
  std::vector<FLOATING_TYPE> c_values = { 0, 1 },
    values;
  std::vector<INDEXING_TYPE> c_row_indicies = { 5, 4 },
    c_column_indicies = { 3, 2 },
    row_indicies,
    column_indicies;
  MatrixMarketHeader mmh(Matrix, Sparse, Pattern, General);
  WriteHeader(mmh);
  WriteIndicesPattern(c_rows, c_columns, c_non_zeros, c_row_indicies, c_column_indicies);
  MatrixMarket mm(mmh);

  mm.ReadIndices(ss, rows, columns, non_zeros,
                 row_indicies, column_indicies, values);

  ASSERT_EQ(c_rows, rows);
  ASSERT_EQ(c_columns, columns);
  ASSERT_EQ(c_non_zeros, non_zeros);
  ASSERT_EQ(c_non_zeros, row_indicies.size());
  ASSERT_EQ(c_non_zeros, column_indicies.size());
  ASSERT_EQ(c_non_zeros, values.size());
  assertArrays(&c_row_indicies[0], &row_indicies[0], c_non_zeros, "");
  assertArrays(&c_column_indicies[0], &column_indicies[0], c_non_zeros, "");
}