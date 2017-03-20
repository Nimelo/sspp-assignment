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
      CSR();
      CSR(const INDEXING_TYPE non_zeros, const INDEXING_TYPE rows, const INDEXING_TYPE columns,
          std::vector<INDEXING_TYPE> * irp, std::vector<INDEXING_TYPE> * ja, std::vector<FLOATING_TYPE> * as);
      ~CSR();

      INDEXING_TYPE GetNonZeros() const;
      INDEXING_TYPE GetRows() const;
      INDEXING_TYPE GetColumns() const;
      std::vector<INDEXING_TYPE> * GetIRP();
      std::vector<INDEXING_TYPE> * GetJA();
      std::vector<FLOATING_TYPE> * GetAS();

      friend std::ostream & operator<<(std::ostream & os, const CSR & csr);
      friend std::istream & operator >> (std::istream & is, CSR & csr);
    protected:
      INDEXING_TYPE non_zeros_;
      INDEXING_TYPE rows_;
      INDEXING_TYPE columns_;
      std::vector<INDEXING_TYPE> *irp_;
      std::vector<INDEXING_TYPE> *ja_;
      std::vector<FLOATING_TYPE> *as_;
    };
  }
}

#endif
