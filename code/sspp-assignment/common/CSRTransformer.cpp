#include "CSRTransformer.h"
#include "Definitions.h"
#include "InPlaceStableSorter.h"

sspp::representations::CSR sspp::tools::transformers::CSRTransformer::Transform(representations::IntermediarySparseMatrix & ism) {
  tools::sorters::InPlaceStableSorter sorter;
  sorter.Sort(&ism.GetRowIndexes()[0], &ism.GetColumnIndexes()[0], &ism.GetValues()[0], ism.GetNonZeros());

  std::vector<FLOATING_TYPE> as(ism.GetNonZeros());
  INDEXING_TYPE index = 0;
  std::vector<INDEXING_TYPE> irp(ism.GetRows() + 1),
    ja(ism.GetNonZeros());

  as[0] = ism.GetValues()[0];
  ja[0] = ism.GetColumnIndexes()[0];
  irp[0] = 0;

  for(auto i = 1; i < ism.GetNonZeros(); i++) {
    as[i] = ism.GetValues()[i];
    ja[i] = ism.GetColumnIndexes()[i];
    if(ism.GetRowIndexes()[i - 1] != ism.GetRowIndexes()[i])
      irp[++index] = i;
  }

  irp[++index] = ism.GetNonZeros();

  return representations::CSR(ism.GetNonZeros(), ism.GetRows(), ism.GetColumns(), irp, ja, as);
}
