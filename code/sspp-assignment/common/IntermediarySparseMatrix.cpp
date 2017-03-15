#include "IntermediarySparseMatrix.h"

sspp::representations::IntermediarySparseMatrix::IntermediarySparseMatrix() {
}

sspp::representations::IntermediarySparseMatrix::IntermediarySparseMatrix(int m, int n, int nz, int * iIndexes, int * jIndexes, FLOATING_TYPE * values)
  : NZ(nz), M(m), N(n), IIndexes(iIndexes), JIndexes(jIndexes), Values(values) {
}

sspp::representations::IntermediarySparseMatrix::IntermediarySparseMatrix(const IntermediarySparseMatrix & other) {
  this->N = other.N;
  this->M = other.M;
  this->NZ = other.NZ;

  this->IIndexes = new int[NZ];
  this->JIndexes = new int[NZ];
  this->Values = new FLOATING_TYPE[NZ];

  for(auto it = 0; it < NZ; it++) {
    IIndexes[it] = other.IIndexes[it];
    JIndexes[it] = other.JIndexes[it];
    Values[it] = other.Values[it];
  }
}

sspp::representations::IntermediarySparseMatrix & sspp::representations::IntermediarySparseMatrix::operator=(representations::IntermediarySparseMatrix rhs) {
  this->N = rhs.N;
  this->M = rhs.M;
  this->NZ = rhs.NZ;

  this->IIndexes = new int[NZ];
  this->JIndexes = new int[NZ];
  this->Values = new FLOATING_TYPE[NZ];

  for(auto it = 0; it < NZ; it++) {
    IIndexes[it] = rhs.IIndexes[it];
    JIndexes[it] = rhs.JIndexes[it];
    Values[it] = rhs.Values[it];
  }

  return *this;
}

sspp::representations::IntermediarySparseMatrix::~IntermediarySparseMatrix() {
  if(IIndexes != 0)
    delete[] IIndexes;
  if(JIndexes != 0)
    delete[] JIndexes;
  if(Values != 0)
    delete[] Values;
}