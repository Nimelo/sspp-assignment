#ifndef SSPP_COMMON_CSRTRANSFORMER_H_
#define SSPP_COMMON_CSRTRANSFORMER_H_

#include "CSR.h"
#include "IntermediarySparseMatrix.h"

namespace sspp {
  namespace tools {
    namespace transformers {
      class CSRTransformer {
      public:
        representations::CSR transform(representations::IntermediarySparseMatrix & ism);
      };
    }
  }
}

#endif
