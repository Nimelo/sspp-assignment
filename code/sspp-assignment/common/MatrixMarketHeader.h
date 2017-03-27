#ifndef SSPP_COMMON_MATRIXMARKETHEAER_H_
#define SSPP_COMMON_MATRIXMARKETHEAER_H_

#include "MatrixMarketEnums.h"
#include <istream>
#include <map>

namespace sspp {
  namespace common {
    class MatrixMarketHeader {
    public:
      MatrixMarketHeader(MatrixMarketCode code, MatrixMarketFormat format,
                         MatrixMarketDataType data_type, MatrixMarketStorageScheme storage_scheme);
      MatrixMarketHeader() = default;

      bool IsMatrix() const;
      bool IsSparse() const;
      bool IsDense() const;
      bool IsComplex() const;
      bool IsReal() const;
      bool IsPattern() const;
      bool IsInteger() const;
      bool IsSymmetric() const;
      bool IsGeneral() const;
      bool IsSkew() const;
      bool IsHermitian() const;
      bool IsValid() const;

      std::string ToString() const;
      static std::map<std::string, MatrixMarketFormat> GetFormatsMap();
      static std::map<std::string, MatrixMarketDataType> GetDataTypesMap();
      static std::map<std::string, MatrixMarketStorageScheme> GetStorageSchemesMap();
      friend std::istream& operator>>(std::istream& is, MatrixMarketHeader& mmh);

    private:
      MatrixMarketCode code_;
      MatrixMarketFormat format_;
      MatrixMarketDataType data_type_;
      MatrixMarketStorageScheme storage_scheme_;
    public:
      static constexpr const char MM_MTX_STR[] = "matrix";
      static constexpr const char MM_DENSE_STR[] = "array";
      static constexpr const char MM_SPARSE_STR[] = "coordinate";
      static constexpr const char MM_COMPLEX_STR[] = "complex";
      static constexpr const char MM_REAL_STR[] = "real";
      static constexpr const char MM_INT_STR[] = "integer";
      static constexpr const char MM_GENERAL_STR[] = "general";
      static constexpr const char MM_SYMM_STR[] = "symmetric";
      static constexpr const char MM_HERM_STR[] = "hermitian";
      static constexpr const char MM_SKEW_STR[] = "skew-symmetric";
      static constexpr const char MM_PATTERN_STR[] = "pattern";
      static constexpr const char MatrixMarketBanner_STR[] = "%%MatrixMarket";
    };
  }
}
#endif
