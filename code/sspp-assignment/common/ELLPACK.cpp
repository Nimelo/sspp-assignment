#include "ELLPACK.h"

sspp::representations::ELLPACK::ELLPACK()
  :rows_{ 0 },
  columns_{ 0 },
  non_zeros_{ 0 },
  max_row_non_zeros_{ 0 },
  ja_{ new std::vector<INDEXING_TYPE> },
  as_{ new std::vector<FLOATING_TYPE> } {
}

sspp::representations::ELLPACK::~ELLPACK() {
  if(ja_ != nullptr)
    delete ja_;
  if(as_ != nullptr)
    delete as_;
}

sspp::representations::ELLPACK::ELLPACK(const INDEXING_TYPE rows, const INDEXING_TYPE columns, const INDEXING_TYPE non_zeros,
                                        const INDEXING_TYPE max_row_non_zeros, std::vector<INDEXING_TYPE> *ja,
                                        std::vector<FLOATING_TYPE> *as)
  :rows_{ rows },
  columns_{ columns },
  non_zeros_{ non_zeros },
  max_row_non_zeros_{ max_row_non_zeros },
  ja_{ ja },
  as_{ as } {
}

INDEXING_TYPE sspp::representations::ELLPACK::CalculateIndex(INDEXING_TYPE row, INDEXING_TYPE column) const {
  return row * max_row_non_zeros_ + column;
}

INDEXING_TYPE sspp::representations::ELLPACK::GetRows() const {
  return rows_;
}

INDEXING_TYPE sspp::representations::ELLPACK::GetColumns() const {
  return columns_;
}

INDEXING_TYPE sspp::representations::ELLPACK::GetNonZeros() const {
  return non_zeros_;
}

INDEXING_TYPE sspp::representations::ELLPACK::GetMaxRowNonZeros() const {
  return max_row_non_zeros_;
}

std::vector<INDEXING_TYPE> * sspp::representations::ELLPACK::GetJA() {
  return ja_;
}

std::vector<FLOATING_TYPE> * sspp::representations::ELLPACK::GetAS() {
  return as_;
}

std::ostream & sspp::representations::operator<<(std::ostream & os, const ELLPACK & ellpack) {
  os << ellpack.rows_ << LINE_SEPARATOR;
  os << ellpack.columns_ << LINE_SEPARATOR;
  os << ellpack.non_zeros_ << LINE_SEPARATOR;
  os << ellpack.max_row_non_zeros_ << LINE_SEPARATOR;

  for(unsigned int i = 0; i < ellpack.ja_->size() - 1; i++) {
    os << (*ellpack.ja_)[i] << SPACE;
  }
  os << (*ellpack.ja_)[ellpack.ja_->size() - 1] << LINE_SEPARATOR;

  for(unsigned int i = 0; i < ellpack.as_->size() - 1; i++) {
    os << (*ellpack.as_)[i] << SPACE;
  }
  os << (*ellpack.as_)[ellpack.as_->size() - 1] << LINE_SEPARATOR;

  return os;
}

std::istream & sspp::representations::operator >> (std::istream & is, ELLPACK & ellpack) {
  is >> ellpack.rows_;
  is >> ellpack.columns_;
  is >> ellpack.non_zeros_;
  is >> ellpack.max_row_non_zeros_;

  ellpack.ja_->resize(ellpack.rows_ * ellpack.max_row_non_zeros_);
  ellpack.as_->resize(ellpack.rows_ * ellpack.max_row_non_zeros_);

  for(auto &value : *ellpack.ja_) {
    is >> value;
  }

  for(auto &value : *ellpack.as_) {
    is >> value;
  }

  return is;
}
