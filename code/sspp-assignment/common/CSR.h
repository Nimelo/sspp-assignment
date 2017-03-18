#ifndef SSPP_COMMON_CSR_H_
#define SSPP_COMMON_CSR_H_

#include "Definitions.h"
#include <istream>
#include <ostream>
#include <vector>

namespace sspp {
  namespace representations {
    class CSR {
    public:
      CSR() = default;
      ~CSR() = default;

      CSR(const INDEXING_TYPE non_zeros, const INDEXING_TYPE rows, const INDEXING_TYPE columns,
          const std::vector<INDEXING_TYPE> & irp, const std::vector<INDEXING_TYPE> & ja, const std::vector<FLOATING_TYPE> & as);
      CSR(const CSR & other);
      CSR & operator=(const CSR & rhs);

      INDEXING_TYPE GetNonZeros() const;
      INDEXING_TYPE GetRows() const;
      INDEXING_TYPE GetColumns() const;
      std::vector<long> GetIRP() const;
      std::vector<long> GetJA() const;
      std::vector<FLOATING_TYPE> GetAS() const;

      friend std::ostream & operator<<(std::ostream & os, const CSR & csr);
      friend std::istream & operator >> (std::istream & is, CSR & csr);
    protected:
      void rewrite(CSR & lhs, const CSR & rhs);

      INDEXING_TYPE non_zeros_;
      INDEXING_TYPE rows_;
      INDEXING_TYPE columns_;
      std::vector<INDEXING_TYPE> irp_;
      std::vector<INDEXING_TYPE> ja_;
      std::vector<FLOATING_TYPE> as_;
    };
  }
}

#endif
