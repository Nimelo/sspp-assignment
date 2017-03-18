#ifndef SSPP_COMMON_ELLPACKTRANSFORER_H_
#define SSPP_COMMON_ELLPACKTRANSFORER_H_

#include "IntermediarySparseMatrix.h"
#include "ELLPACK.h"
#include "Definitions.h"
#include <vector>

namespace sspp {
  namespace tools {
    namespace transformers {
      class ELLPACKTransformer {
      public:
        representations::ELLPACK Transform(representations::IntermediarySparseMatrix & ism);
      protected:
        void PreprocessISM(representations::IntermediarySparseMatrix & ism);
        std::vector<INDEXING_TYPE> FindAuxilliaryArray(representations::IntermediarySparseMatrix & ism);
        representations::ELLPACK TransformInternal(representations::IntermediarySparseMatrix & ism, INDEXING_TYPE rows,
                                                   INDEXING_TYPE max_row_non_zeros, std::vector<INDEXING_TYPE> & ja,
                                                   std::vector<FLOATING_TYPE> & as, std::vector<INDEXING_TYPE> & auxilliary_array);
      };
    }
  }
}

#endif