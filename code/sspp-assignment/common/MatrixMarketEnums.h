#ifndef SSPP_COMMON_MATRIXMARKETENUMS_H_
#define SSPP_COMMON_MATRIXMARKETENUMS_H_
namespace sspp {
  namespace io {
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
    enum MatrixMarketErrorCodes {
      SUCCESS = 0,
      MM_COULD_NOT_READ_FILE = 11,
      MM_PREMATURE_EOF = 12,
      MM_NOT_MTX = 13,
      MM_NO_HEADER = 14,
      MM_UNSUPPORTED_TYPE = 15,
      MM_LINE_TOO_LONG = 16,
      MM_COULD_NOT_WRITE_FILE = 17,
    };
  }
}
#endif
