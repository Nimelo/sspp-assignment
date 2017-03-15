#ifndef SSPP_COMMON_INETERMEDIARYSPARSEMATRIX
#define SSPP_COMMON_INETERMEDIARYSPARSEMATRIX

#include "Definitions.h"

namespace sspp {
  namespace representations {
    class IntermediarySparseMatrix {
    public:
      int NZ;
      int M;
      int N;
      int *IIndexes;
      int *JIndexes;
      FLOATING_TYPE *Values;
    public:
      IntermediarySparseMatrix();
      IntermediarySparseMatrix(int m, int n, int nz, int *iIndexes, int *jIndexes, FLOATING_TYPE *values);
      IntermediarySparseMatrix(const IntermediarySparseMatrix &other);
      IntermediarySparseMatrix & operator=(IntermediarySparseMatrix rhs);
      ~IntermediarySparseMatrix();
    };
  }
}

#endif
