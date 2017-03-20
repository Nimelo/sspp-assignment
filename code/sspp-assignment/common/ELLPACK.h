#ifndef SSPP_COMMON_ELLPACK_H_
#define SSPP_COMMON_ELLPACK_H_

#include "Definitions.h"
#include <istream>
#include <ostream>
#include <vector>

namespace sspp {
  namespace representations {
    class ELLPACK {
    public:
      ELLPACK();
      ~ELLPACK();

      ELLPACK(const INDEXING_TYPE rows,
              const INDEXING_TYPE columns,
              const INDEXING_TYPE non_zeros,
              const INDEXING_TYPE max_row_non_zeros,
              std::vector<INDEXING_TYPE> *ja,
              std::vector<FLOATING_TYPE> *as);

      INDEXING_TYPE CalculateIndex(INDEXING_TYPE row, INDEXING_TYPE column) const;
      INDEXING_TYPE GetRows() const;
      INDEXING_TYPE GetColumns() const;
      INDEXING_TYPE GetNonZeros() const;
      INDEXING_TYPE GetMaxRowNonZeros() const;
      std::vector<INDEXING_TYPE> * GetJA();
      std::vector<FLOATING_TYPE> * GetAS();

      friend std::ostream & operator<<(std::ostream & os, const ELLPACK & ellpack);
      friend std::istream & operator >> (std::istream & is, ELLPACK & ellpack);
    protected:
      INDEXING_TYPE rows_;
      INDEXING_TYPE columns_;
      INDEXING_TYPE non_zeros_;
      INDEXING_TYPE max_row_non_zeros_;
      std::vector<INDEXING_TYPE> *ja_;
      std::vector<FLOATING_TYPE> *as_;
    };
  }
}

#endif
