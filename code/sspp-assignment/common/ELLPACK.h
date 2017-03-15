#ifndef SSPP_COMMON_ELLPACK_H_
#define SSPP_COMMON_ELLPACK_H_

#include "Definitions.h"
#include <istream>
#include <ostream>

namespace sspp {
  namespace representations {
    class ELLPACK {
    public:
      int M;
      int N;
      int NZ;
      int MAXNZ;
      int **JA;
      FLOATING_TYPE **AS;
    protected:
      static void rewrite(ELLPACK & lhs, const ELLPACK & rhs);
    public:
      ELLPACK();
      ELLPACK(int M, int N, int NZ, int MAXNZ, int **JA, FLOATING_TYPE **AS);
      ELLPACK(const ELLPACK & other);
      ELLPACK & operator=(ELLPACK rhs);
      ~ELLPACK();
      friend std::ostream & operator<<(std::ostream & os, const ELLPACK & ellpack);
      friend std::istream & operator >> (std::istream & is, ELLPACK & ellpack);
    };
  }
}

#endif
