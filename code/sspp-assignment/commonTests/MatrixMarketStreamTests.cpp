#include "MatrixMarketStreamTest.h"

#include "MatrixMarketStream.h"
#include "UnsignedFloatReader.h"
#include <gmock/gmock.h>

using namespace sspp::common;

TEST_F(MatrixMarketStreamTest, CRG) {
  unsigned c_rows = 5, c_columns = 5, c_non_zeros = 2;
  std::vector<float> c_values = { 0, 1 },
    values;
  std::vector<unsigned> c_row_indicies = { 5, 4 },
    c_column_indicies = { 3, 2 },
    row_indicies,
    column_indicies;
  MatrixMarketHeader mmh(Matrix, Sparse, Real, General);
  WriteHeader(mmh);
  WriteIndices<unsigned, unsigned, float>(c_rows, c_columns, c_non_zeros, c_row_indicies, c_column_indicies, c_values);
  UnsignedFlaotReader reader;
  MatrixMarketStream<float> mms(ss, reader);

  while(mms.hasNextTuple()) {
    const MatrixMarketTuple<float> tuple = mms.GetNextTuple();
    row_indicies.push_back(tuple.GetRowIndice());
    column_indicies.push_back(tuple.GetColumnIndice());
    values.push_back(tuple.GetValue());
  }

  ASSERT_EQ(c_rows, mms.GetRows());
  ASSERT_EQ(c_columns, mms.GetColumns());
  ASSERT_EQ(c_non_zeros, mms.GetNonZeros());
  ASSERT_EQ(c_non_zeros, row_indicies.size());
  ASSERT_EQ(c_non_zeros, column_indicies.size());
  ASSERT_EQ(c_non_zeros, values.size());

  ASSERT_THAT(c_values, testing::ContainerEq(values));
  ASSERT_THAT(row_indicies, ::testing::ElementsAre(4, 3));
  ASSERT_THAT(column_indicies, ::testing::ElementsAre(2, 1));
}