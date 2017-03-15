#ifndef SSPP_COMMON_ELLPACKTRANSFORER_H_
#define SSPP_COMMON_ELLPACKTRANSFORER_H_

#include "IntermediarySparseMatrix.h"
#include "ELLPACK.h"
#include "Definitions.h"

namespace sspp {
  namespace tools {
    namespace transformers {
      class ELLPACKTransformer {
      public:
        representations::ELLPACK transform(const representations::IntermediarySparseMatrix & ism);
      protected:
        void preprocessISM(const representations::IntermediarySparseMatrix & ism);
        int * findAuxilliaryArray(const representations::IntermediarySparseMatrix & ism);
        void allocateArrays(int ***JA, FLOATING_TYPE ***AS, int M, int MAXNZ);
        representations::ELLPACK transformImpl(const representations::IntermediarySparseMatrix & ism, int M, int MAXNZ, int **JA, FLOATING_TYPE **AS, int *auxArray);
      };
    }
  }
}

#endif