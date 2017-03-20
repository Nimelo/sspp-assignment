#include "ELLPACKTransformer.h"
#include "InPlaceStableSorter.h"
#include <algorithm>

void sspp::tools::transformers::ELLPACKTransformer::PreprocessISM(representations::IntermediarySparseMatrix & ism) {
  tools::sorters::InPlaceStableSorter sorter;
  sorter.Sort(*ism.GetRowIndexes(), *ism.GetColumnIndexes(), *ism.GetValues(), ism.GetNonZeros());
}

std::vector<INDEXING_TYPE> *sspp::tools::transformers::ELLPACKTransformer::FindAuxilliaryArray(representations::IntermediarySparseMatrix & ism) {
  auto aux_array = new std::vector<INDEXING_TYPE>(ism.GetRows());

  auto tmp = 0;
  auto index = 0;
  for(auto i = 1; i < ism.GetNonZeros(); i++) {
    if(ism.GetRowIndexes()->at(i - 1) == ism.GetRowIndexes()->at(i)) {
      ++tmp;
    } else {
      aux_array->at(index++) = ++tmp;
      tmp = 0;
    }
  }

  aux_array->at(index++) = tmp != 0 ? ++tmp : 1;
  return aux_array;
}

sspp::representations::ELLPACK *sspp::tools::transformers::ELLPACKTransformer::TransformInternal(representations::IntermediarySparseMatrix& ism,
                                                                                                 INDEXING_TYPE rows,
                                                                                                 INDEXING_TYPE max_row_non_zeros,
                                                                                                 std::vector<INDEXING_TYPE>* ja,
                                                                                                 std::vector<FLOATING_TYPE>* as,
                                                                                                 std::vector<INDEXING_TYPE>* auxilliary_array) {
  auto nz_index = 0;
  for(auto row = 0; row < rows; row++) {
    for(auto column = 0; column < auxilliary_array->at(row); column++) {
      auto index = row * max_row_non_zeros + column;
      as->at(index) = ism.GetValues()->at(nz_index);
      ja->at(index) = ism.GetColumnIndexes()->at(nz_index);
      ++nz_index;
    }

    if(auxilliary_array->at(row) < max_row_non_zeros) {
      for(auto i = auxilliary_array->at(row); i < max_row_non_zeros; i++) {
        auto index = row * max_row_non_zeros + i;
        as->at(index) = 0;
        if(auxilliary_array->at(row) != 0)
          ja->at(index) = ja->at(index - 1);
        else
          ja->at(index) = 0;
      }
    }
  }

  return new representations::ELLPACK(rows, ism.GetColumns(), ism.GetNonZeros(), max_row_non_zeros, ja, as);
}

sspp::representations::ELLPACK *sspp::tools::transformers::ELLPACKTransformer::Transform(representations::IntermediarySparseMatrix & ism) {
  PreprocessISM(ism);
  auto aux = FindAuxilliaryArray(ism);
  auto MAXNZ = *std::max_element(aux->begin(), aux->end() - 1);
  auto ja = new std::vector<INDEXING_TYPE>(MAXNZ * ism.GetRows());
  auto as = new std::vector<FLOATING_TYPE>(MAXNZ * ism.GetRows());

  auto ellpack = TransformInternal(ism, ism.GetRows(), MAXNZ, ja, as, aux);
  delete aux;
  return ellpack;
}
