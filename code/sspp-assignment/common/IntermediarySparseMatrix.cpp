#include "IntermediarySparseMatrix.h"

sspp::representations::IntermediarySparseMatrix::IntermediarySparseMatrix(const INDEXING_TYPE rows, const INDEXING_TYPE columns,
                                                                          const INDEXING_TYPE non_zeros,
                                                                          const std::vector<INDEXING_TYPE>& row_indexes,
                                                                          const std::vector<INDEXING_TYPE>& column_indexes,
                                                                          const std::vector<FLOATING_TYPE>& values)
  : non_zeros_{ non_zeros }, rows_{ rows }, columns_{ columns }, row_indexes_{ row_indexes }, column_indexes_{ column_indexes }, values_{ values } {
}

sspp::representations::IntermediarySparseMatrix::IntermediarySparseMatrix(const IntermediarySparseMatrix & other) {
  Rewrite(*this, other);
}

sspp::representations::IntermediarySparseMatrix& sspp::representations::IntermediarySparseMatrix::operator=(const IntermediarySparseMatrix& rhs) {
  Rewrite(*this, rhs);
  return *this;
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

std::vector<INDEXING_TYPE> sspp::representations::IntermediarySparseMatrix::GetRowIndexes() const {
  return row_indexes_;
}

std::vector<INDEXING_TYPE> sspp::representations::IntermediarySparseMatrix::GetColumnIndexes() const {
  return column_indexes_;
}

std::vector<FLOATING_TYPE> sspp::representations::IntermediarySparseMatrix::GetValues() const {
  return values_;
}

void sspp::representations::IntermediarySparseMatrix::Rewrite(IntermediarySparseMatrix& lhs, const IntermediarySparseMatrix& rhs) {
  lhs.non_zeros_ = rhs.non_zeros_;
  lhs.columns_ = rhs.columns_;
  lhs.rows_ = rhs.rows_;
  lhs.row_indexes_ = std::vector<INDEXING_TYPE>(rhs.row_indexes_);
  lhs.column_indexes_ = std::vector<INDEXING_TYPE>(rhs.column_indexes_);
  lhs.values_ = std::vector<FLOATING_TYPE>(rhs.values_);
}
