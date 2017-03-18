#include "MatrixMarket.h"
#include <string>
#include <sstream>
#include <ctime>

sspp::io::MatrixMarket::MatrixMarket(MatrixMarketHeader header)
  :header_(header) {
}

sspp::io::MatrixMarketErrorCodes sspp::io::MatrixMarket::ReadIndices(std::istream& is,
                                                                     INDEXING_TYPE & rows,
                                                                     INDEXING_TYPE & columns,
                                                                     INDEXING_TYPE & non_zeros,
                                                                     std::vector<INDEXING_TYPE>& row_indices,
                                                                     std::vector<INDEXING_TYPE>& column_indices,
                                                                     std::vector<FLOATING_TYPE>& values) {
  if(!header_.IsValid()
     || header_.IsComplex()
     || header_.IsHermitian()
     || header_.IsSkew())
    return MatrixMarketErrorCodes::MM_UNSUPPORTED_TYPE;

  std::string line;

  while(!is.eof()) {
    std::getline(is, line);
    if(line.find_first_of(COMMENT_PREFIX_STR) != 0) {
      std::stringstream ss(line);
      ss >> rows >> columns >> non_zeros;
      return ReadIndices(is, non_zeros, row_indices, column_indices, values);
    }
  }
}

sspp::io::MatrixMarketErrorCodes sspp::io::MatrixMarket::ReadIndices(std::istream &is,
                                                                     INDEXING_TYPE & non_zeros,
                                                                     std::vector<INDEXING_TYPE>& row_indices,
                                                                     std::vector<INDEXING_TYPE>& column_indices,
                                                                     std::vector<FLOATING_TYPE>& values) {
  INDEXING_TYPE elements = non_zeros;
  if(header_.IsSymmetric()) {
    non_zeros <<= 1;
  }

  row_indices.resize(non_zeros);
  column_indices.resize(non_zeros);
  values.resize(non_zeros);

  srand(time(0));
  if(header_.IsPattern()) {
    for(INDEXING_TYPE i = 0; i < elements; i++) {
      is >> row_indices[i] >> column_indices[i];
      values[i] = rand() % RAND_LIMIT;
    }
  } else {
    for(INDEXING_TYPE i = 0; i < elements; i++) {
      is >> row_indices[i] >> column_indices[i] >> values[i];
    }
  }

  if(header_.IsSymmetric()) {
    for(INDEXING_TYPE i = 0; i < elements; i++) {
      auto index = i + elements;
      row_indices[index] = column_indices[i];
      column_indices[index] = row_indices[i];
      values[index] = values[i];
    }
  }
  return SUCCESS;
}
