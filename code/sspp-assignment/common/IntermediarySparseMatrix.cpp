#include "IntermediarySparseMatrix.h"

sspp::representations::IntermediarySparseMatrix::IntermediarySparseMatrix(const INDEXING_TYPE rows,
                                                                          const INDEXING_TYPE columns,
                                                                          const INDEXING_TYPE non_zeros,
                                                                          std::vector<INDEXING_TYPE>* row_indexes,
                                                                          std::vector<INDEXING_TYPE>* column_indexes,
                                                                          std::vector<FLOATING_TYPE>* values)
  : non_zeros_{ non_zeros },
  rows_{ rows },
  columns_{ columns },
  row_indexes_{ row_indexes },
  column_indexes_{ column_indexes },
  values_{ values } {
}

sspp::representations::IntermediarySparseMatrix::~IntermediarySparseMatrix() {
  if(row_indexes_ != nullptr)
    delete row_indexes_;
  if(column_indexes_ != nullptr)
    delete column_indexes_;
  if(values_ != nullptr)
    delete values_;
}

INDEXING_TYPE sspp::representations::IntermediarySparseMatrix::GetRows() const {
  return rows_;
}

INDEXING_TYPE sspp::representations::IntermediarySparseMatrix::GetColumns() const {
  return columns_;
}

INDEXING_TYPE sspp::representations::IntermediarySparseMatrix::GetNonZeros() const {
  return non_zeros_;
}

std::vector<INDEXING_TYPE> * sspp::representations::IntermediarySparseMatrix::GetRowIndexes() {
  return row_indexes_;
}

std::vector<INDEXING_TYPE> * sspp::representations::IntermediarySparseMatrix::GetColumnIndexes() {
  return column_indexes_;
}

std::vector<FLOATING_TYPE> * sspp::representations::IntermediarySparseMatrix::GetValues() {
  return values_;
}
