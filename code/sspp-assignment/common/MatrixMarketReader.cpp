#include "MatrixMarketReader.h"
#include "InPlaceStableSorter.h"
#include "ReadMatrixException.h"
#include "mmio.h"

sspp::representations::IntermediarySparseMatrix sspp::io::readers::MatrixMarketReader::FromFile(std::string file_name) {
  int M, N, nz, *I, *J;
  FLOATING_TYPE *val;

  int result = mm_read_sparse(file_name.c_str(), &M, &N, &nz, &val, &I, &J);

  return sspp::representations::IntermediarySparseMatrix(M, N, nz, std::vector<INDEXING_TYPE>(I, I + nz),
                                                         std::vector<INDEXING_TYPE>(J, J + nz),
                                                         std::vector<FLOATING_TYPE>(val, val + nz));
}
