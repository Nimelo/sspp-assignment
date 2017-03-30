#ifndef SSPP_COMMON_MATRIXMARKETTUPLE_H_
#define SSPP_COMMON_MATRIXMARKETTUPLE_H_

namespace sspp {
  namespace common {
    template<typename VALUE_TYPE>
    class MatrixMarketTuple {
    public:
      MatrixMarketTuple() :row_indice_(0), column_indice_(0) {};

      MatrixMarketTuple(unsigned row_indice, unsigned column_indice, VALUE_TYPE value)
        :row_indice_(row_indice), column_indice_(column_indice), value_(value) {
      };

      MatrixMarketTuple(const MatrixMarketTuple<VALUE_TYPE> & other) {
        Swap(*this, other);
      }

      template<typename T>
      MatrixMarketTuple<T> & operator=(const MatrixMarketTuple<T> & rhs) {
        Swap(this, rhs);
        return *this;
      }

      unsigned GetRowIndice() const {
        return row_indice_;
      };

      unsigned GetColumnIndice() const {
        return column_indice_;
      };

      VALUE_TYPE GetValue() const {
        return value_;
      };
    private:
      static void Swap(MatrixMarketTuple<VALUE_TYPE> & lhs, const MatrixMarketTuple<VALUE_TYPE> & rhs) {
        lhs.row_indice_ = rhs.row_indice_;
        lhs.column_indice_ = rhs.column_indice_;
        lhs.value_ = rhs.value_;
      }

      unsigned row_indice_;
      unsigned column_indice_;
      VALUE_TYPE value_;
    };
  }
}
#endif
