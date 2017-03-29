#ifndef SSPP_COMMON_CSR_H_
#define SSPP_COMMON_CSR_H_

#include "MatrixMarketStream.h"
#include "MatrixMarketHeader.h"
#include "StableSorter.h"

#include <istream>
#include <ostream>
#include <vector>

namespace sspp {
  namespace common {
    template<typename VALUE_TYPE>
    class CRS {
    public:
      CRS() :
        non_zeros_(0),
        rows_(0),
        columns_(0) {

      };

      explicit CRS(MatrixMarketStream<VALUE_TYPE> & mms) {
        const MatrixMarketHeader header = mms.GetMatrixMarketHeader();
        if(header.IsValid()) {
          Load(mms);
        }
      };

      CRS(const unsigned non_zeros,
          const unsigned rows,
          const unsigned columns,
          const std::vector<unsigned> & irp,
          const std::vector<unsigned> & ja,
          const std::vector<VALUE_TYPE> & as) :
        non_zeros_(non_zeros),
        rows_(rows),
        columns_(columns),
        row_start_indexes_(irp),
        column_indices_(ja),
        values_(as) {
      };

      unsigned GetNonZeros() const {
        return non_zeros_;
      };

      unsigned GetRows() const {
        return rows_;
      }

      unsigned GetColumns() const {
        return columns_;
      };

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
        os << crs.rows_ << std::endl;
        os << crs.columns_ << std::endl;
        os << crs.non_zeros_ << std::endl;

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
      void Load(MatrixMarketStream<VALUE_TYPE> & mms) {
        const MatrixMarketHeader header = mms.GetMatrixMarketHeader();
        rows_ = mms.GetRows();
        columns_ = mms.GetColumns();
        non_zeros_ = mms.GetNonZeros();

        if(header.IsSymmetric()) {
          LoadSymetric(mms);
        } else {
          LoadGeneral(mms);
        }
      };

      void LoadSymetric(MatrixMarketStream<VALUE_TYPE> & mms) {
        unsigned expected_non_zeros = mms.GetNonZeros() << 1;
        std::vector<unsigned> row_indices(expected_non_zeros);
        column_indices_.resize(expected_non_zeros);
        values_.resize(expected_non_zeros);
        unsigned non_zeros = 0;
        while(mms.hasNextTuple()) {
          MatrixMarketTuple<VALUE_TYPE> tuple = mms.GetNextTuple();
          row_indices[non_zeros] = tuple.GetRowIndice();
          column_indices_[non_zeros] = tuple.GetColumnIndice();
          values_[non_zeros] = tuple.GetValue();
          ++non_zeros;

          if(tuple.GetColumnIndice() != tuple.GetRowIndice()) {
            row_indices[non_zeros] = tuple.GetColumnIndice();
            column_indices_[non_zeros] = tuple.GetRowIndice();
            values_[non_zeros] = tuple.GetValue();
            ++non_zeros;
          }
        }

        non_zeros_ = non_zeros;
        SetCSRBasedOnIndices(row_indices);
        //TODO: Check performance issues.
        column_indices_.resize(non_zeros_);
        values_.resize(non_zeros_);
      }

      void LoadGeneral(MatrixMarketStream<VALUE_TYPE> & mms) {
        std::vector<unsigned> row_indices(mms.GetNonZeros());
        column_indices_.resize(mms.GetNonZeros());
        values_.resize(mms.GetNonZeros());

        for(unsigned i = 0; i < mms.GetNonZeros(); ++i) {
          if(mms.hasNextTuple()) {
            MatrixMarketTuple<VALUE_TYPE> tuple = mms.GetNextTuple();
            row_indices[i] = tuple.GetRowIndice();
            column_indices_[i] = tuple.GetColumnIndice();
            values_[i] = tuple.GetValue();
          }
        }
        SetCSRBasedOnIndices(row_indices);
      }

      void SetCSRBasedOnIndices(std::vector<unsigned> & row_indices) {
        StableSorter::InsertionSort<VALUE_TYPE>(row_indices, column_indices_, values_, non_zeros_);
        row_start_indexes_.resize(rows_ + 1);
        unsigned row_start_indexes_index = 0, non_zeros_index = 1;
        row_start_indexes_[row_start_indexes_index++] = 0;

        while(row_start_indexes_index < rows_ + 1
              && non_zeros_index < non_zeros_) {
          unsigned row_indices_diff = row_indices[non_zeros_index] - row_indices[non_zeros_index - 1];
          if(row_indices_diff != 0) {
            row_start_indexes_[row_start_indexes_index++] = non_zeros_index;
            if(row_indices_diff > 1) {
              for(unsigned i = 0; i < row_indices_diff - 1; ++i) {
                row_start_indexes_[row_start_indexes_index++] = non_zeros_index;
              }
            }
          }
          ++non_zeros_index;
        }
        for(unsigned i = row_start_indexes_index; i < rows_; ++i) {
          row_start_indexes_[row_start_indexes_index++] = non_zeros_index;
        }
        row_start_indexes_[row_start_indexes_index++] = non_zeros_index;
      };

      unsigned non_zeros_;
      unsigned rows_;
      unsigned columns_;
      std::vector<unsigned> row_start_indexes_;
      std::vector<unsigned> column_indices_;
      std::vector<VALUE_TYPE> values_;
    };
  }
}

#endif
