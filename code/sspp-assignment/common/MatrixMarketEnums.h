#ifndef SSPP_COMMON_MATRIXMARKETENUMS_H_
#define SSPP_COMMON_MATRIXMARKETENUMS_H_
namespace sspp {
  namespace common {
    enum MatrixMarketCode {
      Matrix = 1
    };

    enum MatrixMarketFormat {
      Sparse = 1,
      Dense
    };

    enum MatrixMarketDataType {
      Real = 1,
      Complex,
      Pattern,
      Integer
    };

    enum MatrixMarketStorageScheme {
      General = 1,
      Hermitian,
      Symetric,
      Skew
    };
  }
}
#endif
