#ifndef SSPP_COMMON_ELLPACK_H_
#define SSPP_COMMON_ELLPACK_H_

#include "MatrixMarketStream.h"

#include <istream>
#include <ostream>
#include <vector>

namespace sspp {
  namespace common {
    template<typename VALUE_TYPE>
    class ELLPACK {
    public:
      ELLPACK() : rows_(0), columns_(0), non_zeros_(0), max_row_non_zeros_(0) {}
      ELLPACK(MatrixMarketStream<VALUE_TYPE> & mms) {
        const MatrixMarketHeader header = mms.GetMatrixMarketHeader();
        if(header.IsValid()) {
          Load(mms);
        }
      }

      ELLPACK(const unsigned rows,
              const unsigned columns,
              const unsigned non_zeros,
              const unsigned max_row_non_zeros,
              const std::vector<unsigned> & column_indices,
              const std::vector<VALUE_TYPE> & values) :
        rows_(rows),
        columns_(columns),
        non_zeros_(non_zeros),
        max_row_non_zeros_(max_row_non_zeros),
        column_indices_(column_indices),
        values_(values) {
      };

      unsigned CalculateIndex(unsigned row, unsigned column) const {
        return row + max_row_non_zeros_ + column;
      };

      unsigned GetRows() const {
        return rows_;
      };

      unsigned GetColumns() const {
        return columns_;
      };

      unsigned GetNonZeros() const {
        return non_zeros_;
      };

      unsigned GetMaxRowNonZeros() const {
        return max_row_non_zeros_;
      };

      std::vector<unsigned> const & GetColumnIndices() {
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

        for(unsigned int i = 0; i < ellpack.column_indices_.size() - 1; i++) {
          os << ellpack.column_indices_[i] << '\t';
        }
        os << ellpack.column_indices_[ellpack.column_indices_.size() - 1] << std::endl;;

        for(unsigned int i = 0; i < ellpack.values_.size() - 1; i++) {
          os << ellpack.values_[i] << '\t';
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
        for(std::vector<unsigned>::iterator it = ellpack.column_indices_.begin(); it != ellpack.column_indices_.end(); ++it)
          is >> *it;
        for(typename std::vector<VALUE_TYPE>::iterator it = ellpack.values_.begin(); it != ellpack.values_.end(); ++it)
          is >> *it;

        return is;
      }
    protected:
      void Load(MatrixMarketStream<VALUE_TYPE> & mms) {
        //TODO: Load
      }

      unsigned rows_;
      unsigned columns_;
      unsigned non_zeros_;
      unsigned max_row_non_zeros_;
      std::vector<unsigned> column_indices_;
      std::vector<VALUE_TYPE> values_;
    };
  }
}

#endif
