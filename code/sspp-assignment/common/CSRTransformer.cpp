#include "CSRTransformer.h"
#include "Definitions.h"
#include "InPlaceStableSorter.h"

sspp::representations::CSR sspp::tools::transformers::CSRTransformer::transform(representations::IntermediarySparseMatrix & ism) {
  tools::sorters::InPlaceStableSorter sorter;
  sorter.sort(ism.IIndexes, ism.JIndexes, ism.Values, ism.NZ);

  FLOATING_TYPE *AS = new FLOATING_TYPE[ism.NZ];
  int index = 0, *IRP = new int[ism.M + 1], *JA = new int[ism.NZ];

  AS[0] = ism.Values[0];
  JA[0] = ism.JIndexes[0];
  IRP[0] = 0;

  for(auto i = 1; i < ism.NZ; i++) {
    AS[i] = ism.Values[i];
    JA[i] = ism.JIndexes[i];
    if(ism.IIndexes[i - 1] != ism.IIndexes[i])
      IRP[++index] = i;
  }

  IRP[++index] = ism.NZ;

  return representations::CSR(ism.NZ, ism.M, ism.N, IRP, JA, AS);
}
