#include "CSRTransformer.h"
#include "Definitions.h"
#include "InPlaceStableSorter.h"

sspp::representations::CSR *sspp::tools::transformers::CSRTransformer::Transform(representations::IntermediarySparseMatrix & ism) {
  tools::sorters::InPlaceStableSorter sorter;
  sorter.InsertionSort(*ism.GetRowIndexes(), *ism.GetColumnIndexes(), *ism.GetValues(), 0, ism.GetNonZeros());

  auto as = new std::vector<FLOATING_TYPE>(ism.GetNonZeros());
  INDEXING_TYPE index = 0;
  auto irp = new std::vector<INDEXING_TYPE>(ism.GetRows() + 1),
    ja = new std::vector<INDEXING_TYPE>(ism.GetNonZeros());

  as->at(0) = ism.GetValues()->at(0);
  ja->at(0) = ism.GetColumnIndexes()->at(0);
  irp->at(0) = 0;

  for(auto i = 1; i < ism.GetNonZeros(); i++) {
    as->at(i) = ism.GetValues()->at(i);
    ja->at(i) = ism.GetColumnIndexes()->at(i);
    if(ism.GetRowIndexes()->at(i - 1) != ism.GetRowIndexes()->at(i))
      irp->at(++index) = i;
  }

  irp->at(++index) = ism.GetNonZeros();

  return new representations::CSR(ism.GetNonZeros(), ism.GetRows(), ism.GetColumns(), irp, ja, as);
}
