#include "MatrixMarketReader.h"
#include "InPlaceStableSorter.h"
#include "ReadMatrixException.h"
#include "MatrixMarket.h"
#include <fstream>

sspp::representations::IntermediarySparseMatrix sspp::io::readers::MatrixMarketReader::FromFile(std::string file_name) {
  std::fstream fs;
  fs.open(file_name, std::fstream::in);
  auto result = FromStream(fs);
  fs.close();
  return result;
}

sspp::representations::IntermediarySparseMatrix sspp::io::readers::MatrixMarketReader::FromStream(std::istream& is) {
  INDEXING_TYPE rows, columns, non_zeros;
  std::vector<INDEXING_TYPE> row_indices, column_indices;
  std::vector<FLOATING_TYPE> values;

  MatrixMarketErrorCodes status;
  MatrixMarketHeader mmh;
  status = mmh.Load(is);
  if(status != MatrixMarketErrorCodes::SUCCESS) {
    throw exceptions::ReadMatrixException();
  }
  MatrixMarket mm(mmh);
  is.seekg(0, is.beg);
  mm.ReadIndices(is, rows, columns, non_zeros, row_indices, column_indices, values);

  return sspp::representations::IntermediarySparseMatrix(rows, columns, non_zeros,
                                                         row_indices, column_indices, values);
}
