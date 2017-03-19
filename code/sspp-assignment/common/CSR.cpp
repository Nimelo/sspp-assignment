#include "CSR.h"

void sspp::representations::CSR::rewrite(CSR & lhs, const CSR & rhs) {
  lhs.non_zeros_ = rhs.non_zeros_;
  lhs.rows_ = rhs.rows_;
  lhs.columns_ = rhs.columns_;
  lhs.irp_ = std::vector<INDEXING_TYPE>(rhs.irp_);
  lhs.ja_ = std::vector<INDEXING_TYPE>(rhs.ja_);
  lhs.as_ = std::vector<FLOATING_TYPE>(rhs.as_);
}

sspp::representations::CSR::CSR(const INDEXING_TYPE non_zeros, const INDEXING_TYPE rows, const INDEXING_TYPE columns,
                                const std::vector<INDEXING_TYPE>& irp, const std::vector<INDEXING_TYPE>& ja, const std::vector<FLOATING_TYPE>& as)
  : non_zeros_{ non_zeros }, rows_{ rows }, columns_{ columns }, irp_{ irp }, ja_{ ja }, as_{ as } {
}

sspp::representations::CSR::CSR(const CSR & other) {
  rewrite(*this, other);
}

sspp::representations::CSR& sspp::representations::CSR::operator=(const CSR& rhs) {
  rewrite(*this, rhs);
  return *this;
}

long sspp::representations::CSR::GetNonZeros() const {
  return non_zeros_;
}

long sspp::representations::CSR::GetRows() const {
  return rows_;
}

long sspp::representations::CSR::GetColumns() const {
  return columns_;
}

std::vector<INDEXING_TYPE> & sspp::representations::CSR::GetIRP() {
  return irp_;
}

std::vector<INDEXING_TYPE> & sspp::representations::CSR::GetJA() {
  return ja_;
}

std::vector<FLOATING_TYPE> & sspp::representations::CSR::GetAS() {
  return as_;
}

std::ostream & sspp::representations::operator<<(std::ostream & os, const sspp::representations::CSR & csr) {
  os << csr.rows_ << LINE_SEPARATOR;
  os << csr.columns_ << LINE_SEPARATOR;
  os << csr.non_zeros_ << LINE_SEPARATOR;

  for(unsigned int i = 0; i < csr.irp_.size() - 1; i++)
    os << csr.irp_[i] << SPACE;
  os << csr.irp_[csr.irp_.size() - 1] << LINE_SEPARATOR;

  for(unsigned int i = 0; i < csr.ja_.size() - 1; i++)
    os << csr.ja_[i] << SPACE;
  os << csr.ja_[csr.ja_.size() - 1] << LINE_SEPARATOR;

  for(unsigned int i = 0; i < csr.as_.size() - 1; i++)
    os << csr.as_[i] << SPACE;
  os << csr.as_[csr.as_.size() - 1] << LINE_SEPARATOR;

  return os;
}

std::istream & sspp::representations::operator >> (std::istream & is, sspp::representations::CSR & csr) {
  is >> csr.rows_;
  is >> csr.columns_;
  is >> csr.non_zeros_;

  csr.irp_.resize(csr.rows_ + 1);
  csr.ja_.resize(csr.non_zeros_);
  csr.as_.resize(csr.non_zeros_);

  for(auto &value : csr.irp_)
    is >> value;
  for(auto &value : csr.ja_)
    is >> value;
  for(auto &value : csr.as_)
    is >> value;

  return is;
}
