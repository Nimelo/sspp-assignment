#ifndef SSPP_COMMON_ELLPACK_H_
#define SSPP_COMMON_ELLPACK_H_

#include "SparseMatrix.h"

#include <istream>
#include <ostream>
#include <algorithm>
#include <vector>

namespace sspp {
  namespace common {
    template<typename VALUE_TYPE>
    class ELLPACK : public SparseMatrix {
    public:
      ELLPACK() : SparseMatrix(0, 0, 0), max_row_non_zeros_(0) {}

      ELLPACK(const unsigned long long rows,
              const unsigned long long columns,
              const unsigned long long non_zeros,
              const unsigned long long max_row_non_zeros,
              const std::vector<unsigned long long> & column_indices,
              const std::vector<VALUE_TYPE> & values) :
        SparseMatrix(rows, columns, non_zeros),
        max_row_non_zeros_(max_row_non_zeros),
        column_indices_(column_indices),
        values_(values) {
      };

      unsigned long long CalculateIndex(unsigned long long row, unsigned long long column) const {
        return row * max_row_non_zeros_ + column;
      };

      unsigned long long GetMaxRowNonZeros() const {
        return max_row_non_zeros_;
      };

      std::vector<unsigned long long> const & GetColumnIndices() {
        return column_indices_;
      };

      std::vector<VALUE_TYPE> const & GetValues() {
        return values_;
      };

      friend std::ostream & operator<<(std::ostream & os, const ELLPACK & ellpack) {
        os << ellpack.rows_ << std::endl;
        os << ellpack.columns_ << std::endl;
        os << ellpack.non_zeros_ << std::endl;
        os << ellpack.max_row_non_zeros_ << std::endl;

        for(unsigned long long int i = 0; i < ellpack.column_indices_.size() - 1; i++) {
          os << ellpack.column_indices_[i] << ' ';
        }
        os << ellpack.column_indices_[ellpack.column_indices_.size() - 1] << std::endl;;

        for(unsigned long long int i = 0; i < ellpack.values_.size() - 1; i++) {
          os << ellpack.values_[i] << ' ';
        }
        os << ellpack.values_[ellpack.values_.size() - 1] << std::endl;

        return os;
      }

      friend std::istream & operator >> (std::istream & is, ELLPACK & ellpack) {
        is >> ellpack.rows_;
        is >> ellpack.columns_;
        is >> ellpack.non_zeros_;
        is >> ellpack.max_row_non_zeros_;

        ellpack.column_indices_.resize(ellpack.rows_ * ellpack.max_row_non_zeros_);
        ellpack.values_.resize(ellpack.rows_ * ellpack.max_row_non_zeros_);
        for(std::vector<unsigned long long>::iterator it = ellpack.column_indices_.begin(); it != ellpack.column_indices_.end(); ++it)
          is >> *it;
        for(typename std::vector<VALUE_TYPE>::iterator it = ellpack.values_.begin(); it != ellpack.values_.end(); ++it)
          is >> *it;

        return is;
      }

    protected:
      unsigned long long max_row_non_zeros_;
      std::vector<unsigned long long> column_indices_;
      std::vector<VALUE_TYPE> values_;
    };
  }
}

#endif
