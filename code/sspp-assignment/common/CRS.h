#ifndef SSPP_COMMON_CSR_H_
#define SSPP_COMMON_CSR_H_

#include "MatrixMarketHeader.h"
#include "StableSorter.h"
#include "SparseMatrix.h"

#include <istream>
#include <ostream>
#include <vector>

namespace sspp {
  namespace common {
    template<typename VALUE_TYPE>
    class CRS : public SparseMatrix {
    public:
      CRS() : SparseMatrix(0, 0, 0) {

      };

      CRS(const unsigned rows,
          const unsigned columns,
          const unsigned non_zeros,
          const std::vector<unsigned> & irp,
          const std::vector<unsigned> & ja,
          const std::vector<VALUE_TYPE> & as) :
        SparseMatrix(rows, columns, non_zeros),
        row_start_indexes_(irp),
        column_indices_(ja),
        values_(as) {
      };

      CRS(const CRS<VALUE_TYPE> & other) :
        SparseMatrix(0, 0, 0) {
        Swap(other);
      };

      CRS<VALUE_TYPE> & operator=(const CRS<VALUE_TYPE> & rhs) {
        Swap(rhs);
        return *this;
      }

      std::vector<unsigned> const & GetRowStartIndexes() {
        return row_start_indexes_;
      }

      std::vector<unsigned> const & GetColumnIndices() {
        return column_indices_;
      }

      std::vector<VALUE_TYPE> const & GetValues() {
        return values_;
      }

      friend std::ostream & operator<<(std::ostream & os, const CRS<VALUE_TYPE> & crs) {
        os << crs.rows_ << '\n';
        os << crs.columns_ << '\n';
        os << crs.non_zeros_ << '\n';

        for(unsigned i = 0; i < crs.row_start_indexes_.size() - 1; i++)
          os << crs.row_start_indexes_[i] << ' ';
        os << crs.row_start_indexes_[crs.row_start_indexes_.size() - 1] << std::endl;

        for(unsigned i = 0; i < crs.non_zeros_ - 1; i++)
          os << crs.column_indices_[i] << ' ';
        os << crs.column_indices_[crs.non_zeros_ - 1] << std::endl;

        for(unsigned i = 0; i < crs.non_zeros_ - 1; i++)
          os << crs.values_[i] << ' ';
        os << crs.values_[crs.non_zeros_ - 1] << std::endl;

        return os;
      }

      friend std::istream & operator >> (std::istream & is, CRS<VALUE_TYPE> & crs) {
        is >> crs.rows_;
        is >> crs.columns_;
        is >> crs.non_zeros_;

        crs.row_start_indexes_.resize(crs.rows_ + 1);
        crs.column_indices_.resize(crs.non_zeros_);
        crs.values_.resize(crs.non_zeros_);

        for(unsigned i = 0; i < crs.row_start_indexes_.size(); i++)
          is >> crs.row_start_indexes_[i];
        for(unsigned i = 0; i < crs.column_indices_.size(); i++)
          is >> crs.column_indices_[i];
        for(unsigned i = 0; i < crs.values_.size(); i++)
          is >> crs.values_[i];

        return is;
      }
    protected:

      void Swap(const CRS<VALUE_TYPE> & rhs) {
        this->rows_ = rhs.rows_;
        this->columns_ = rhs.columns_;
        this->non_zeros_ = rhs.non_zeros_;

        this->values_.assign(rhs.values_.begin(), rhs.values_.end());
        this->row_start_indexes_.assign(rhs.row_start_indexes_.begin(), rhs.row_start_indexes_.end());
        this->column_indices_.assign(rhs.column_indices_.begin(), rhs.column_indices_.end());
      }

      std::vector<unsigned> row_start_indexes_;
      std::vector<unsigned> column_indices_;
      std::vector<VALUE_TYPE> values_;
    };
  }
}

#endif
