#ifndef SSPP_COMMON_SPARSEMATRIX_H_
#define SSPP_COMMON_SPARSEMATRIX_H_

namespace sspp {
  namespace common {
    class SparseMatrix {
    public:
      SparseMatrix(unsigned rows, unsigned columns, unsigned non_zeros)
        : rows_(rows), columns_(columns), non_zeros_(non_zeros) {
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

    protected:
      unsigned rows_;
      unsigned columns_;
      unsigned non_zeros_;
    };
  }
}

#endif
