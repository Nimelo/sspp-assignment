#ifndef SSPP_COMMON_SPARSEMATRIX_H_
#define SSPP_COMMON_SPARSEMATRIX_H_

namespace sspp {
  namespace common {
    class SparseMatrix {
    public:
      SparseMatrix(unsigned long long rows, unsigned long long columns, unsigned long long non_zeros)
        : rows_(rows), columns_(columns), non_zeros_(non_zeros) {
      };

      unsigned long long GetNonZeros() const {
        return non_zeros_;
      };

      unsigned long long GetRows() const {
        return rows_;
      }

      unsigned long long GetColumns() const {
        return columns_;
      };

    protected:
      unsigned long long rows_;
      unsigned long long columns_;
      unsigned long long non_zeros_;
    };
  }
}

#endif
