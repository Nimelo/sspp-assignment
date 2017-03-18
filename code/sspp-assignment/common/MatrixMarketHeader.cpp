#include "MatrixMarketHeader.h"
#include <string>
#include <sstream>
#include <algorithm>

sspp::io::MatrixMarketHeader::MatrixMarketHeader(MatrixMarketCode code, MatrixMarketFormat format, MatrixMarketDataType data_type, MatrixMarketStorageScheme storage_scheme)
  :code_(code), format_(format), data_type_(data_type), storage_scheme_(storage_scheme) {
}

bool sspp::io::MatrixMarketHeader::IsMatrix() const {
  return code_ == MatrixMarketCode::Matrix;
}

bool sspp::io::MatrixMarketHeader::IsSparse() const {
  return format_ == Sparse;
}

bool sspp::io::MatrixMarketHeader::IsDense() const {
  return format_ == Dense;
}

bool sspp::io::MatrixMarketHeader::IsComplex() const {
  return data_type_ == Complex;
}

bool sspp::io::MatrixMarketHeader::IsReal() const {
  return data_type_ == Real;
}

bool sspp::io::MatrixMarketHeader::IsPattern() const {
  return data_type_ == Pattern;
}

bool sspp::io::MatrixMarketHeader::IsInteger() const {
  return data_type_ == Integer;
}

bool sspp::io::MatrixMarketHeader::IsSymmetric() const {
  return storage_scheme_ == Symetric;
}

bool sspp::io::MatrixMarketHeader::IsGeneral() const {
  return storage_scheme_ == General;
}

bool sspp::io::MatrixMarketHeader::IsSkew() const {
  return storage_scheme_ == Skew;
}

bool sspp::io::MatrixMarketHeader::IsHermitian() const {
  return storage_scheme_ == Hermitian;
}

bool sspp::io::MatrixMarketHeader::IsValid() const {
  if(IsMatrix()
     && (IsDense() || IsSparse())
     && (IsReal() || IsPattern() || IsInteger() || IsComplex())
     && (IsSymmetric() || IsGeneral() || IsSkew() || IsHermitian()))
    return true;
  else
    return false;
}

std::string sspp::io::MatrixMarketHeader::ToString() const {
  std::stringstream ss;
  ss << MatrixMarketBanner_STR << " "
    << MM_MTX_STR << " "
    << (IsDense() ? MM_DENSE_STR : MM_SPARSE_STR) << " "
    << (IsReal() ? MM_REAL_STR :
        IsComplex() ? MM_COMPLEX_STR :
        IsInteger() ? MM_INT_STR : MM_PATTERN_STR) << " "
    << (IsGeneral() ? MM_GENERAL_STR :
        IsHermitian() ? MM_HERM_STR :
        IsSymmetric() ? MM_SYMM_STR : MM_SKEW_STR);
  return ss.str();
}

sspp::io::MatrixMarketErrorCodes sspp::io::MatrixMarketHeader::Load(std::istream& is) {

  std::string banner, mtx, format, data_type, storage_schema;
  is >> banner >> mtx >> format >> data_type >> storage_schema;
  std::transform(mtx.begin(), mtx.end(), mtx.begin(), ::tolower);
  std::transform(format.begin(), format.end(), format.begin(), ::tolower);
  std::transform(data_type.begin(), data_type.end(), data_type.begin(), ::tolower);
  std::transform(storage_schema.begin(), storage_schema.end(), storage_schema.begin(), ::tolower);

  if(banner != MatrixMarketBanner_STR)
    return MM_NO_HEADER;
  if(mtx != MM_MTX_STR)
    return MM_UNSUPPORTED_TYPE;
  code_ = Matrix;

  auto format_map = GetFormatsMap();
  auto data_types_map = GetDataTypesMap();
  auto storage_schemes_map = GetStorageSchemesMap();

  auto format_value = format_map.find(format);
  if(format_value == format_map.end())
    return MM_UNSUPPORTED_TYPE;
  format_ = format_value->second;
  auto data_type_value = data_types_map.find(data_type);
  if(data_type_value == data_types_map.end())
    return MM_UNSUPPORTED_TYPE;
  data_type_ = data_type_value->second;
  auto storage_scheme_value = storage_schemes_map.find(storage_schema);
  if(storage_scheme_value == storage_schemes_map.end())
    return MM_UNSUPPORTED_TYPE;
  storage_scheme_ = storage_scheme_value->second;

  return SUCCESS;
}

std::map<std::string, sspp::io::MatrixMarketFormat> sspp::io::MatrixMarketHeader::GetFormatsMap() const {
  return{
    {MM_DENSE_STR, Dense},
    {MM_SPARSE_STR, Sparse}
  };
}

std::map<std::string, sspp::io::MatrixMarketDataType> sspp::io::MatrixMarketHeader::GetDataTypesMap() const {
  return{
    {MM_REAL_STR, Real},
    {MM_COMPLEX_STR, Complex},
    {MM_PATTERN_STR, Pattern},
    {MM_INT_STR, Integer}
  };
}

std::map<std::string, sspp::io::MatrixMarketStorageScheme> sspp::io::MatrixMarketHeader::GetStorageSchemesMap() const {
  return{
    {MM_GENERAL_STR, General},
    {MM_HERM_STR, Hermitian},
    {MM_SYMM_STR, Symetric},
    {MM_SKEW_STR, Skew}
  };
}

