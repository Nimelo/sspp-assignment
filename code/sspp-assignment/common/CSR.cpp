#include "CSR.h"

void sspp::representations::CSR::rewrite(CSR & lhs, const CSR & rhs) {
  lhs.NZ = rhs.getASSize();
  lhs.M = rhs.M;
  lhs.N = rhs.N;
  lhs.IRP = new int[rhs.getIRPSize()];
  for(auto i = 0; i < rhs.getIRPSize(); i++)
    lhs.IRP[i] = rhs.IRP[i];

  lhs.JA = new int[rhs.getJASize()];
  for(auto i = 0; i < rhs.getJASize(); i++)
    lhs.JA[i] = rhs.JA[i];

  lhs.AS = new FLOATING_TYPE[rhs.getASSize()];
  for(auto i = 0; i < rhs.getASSize(); i++)
    lhs.AS[i] = rhs.AS[i];
}

sspp::representations::CSR::CSR()
  : NZ(0), M(0), N(0), IRP(0), JA(0), AS(0) {
}

sspp::representations::CSR::CSR(int NZ, int M, int N, int * IRP, int * JA, FLOATING_TYPE * AS)
  : NZ(NZ), M(M), N(N), IRP(IRP), JA(JA), AS(AS) {
}

sspp::representations::CSR::CSR(const CSR & other) {
  rewrite(*this, other);
}

sspp::representations::CSR & sspp::representations::CSR::operator=(CSR rhs) {
  rewrite(*this, rhs);
  return *this;
}

sspp::representations::CSR::~CSR() {
  delete[] IRP;
  delete[] JA;
  delete[] AS;
}

int sspp::representations::CSR::getIRPSize() const {
  return M + 1;
}

int sspp::representations::CSR::getJASize() const {
  return NZ;
}

int sspp::representations::CSR::getASSize() const {
  return NZ;
}

std::ostream & sspp::representations::operator<<(std::ostream & os, const sspp::representations::CSR & csr) {
  os << csr.M << LINE_SEPARATOR;
  os << csr.N << LINE_SEPARATOR;
  os << csr.NZ << LINE_SEPARATOR;

  for(auto i = 0; i < csr.M; i++)
    os << csr.IRP[i] << SPACE;
  os << csr.IRP[csr.M] << LINE_SEPARATOR;

  for(auto i = 0; i < csr.NZ - 1; i++)
    os << csr.JA[i] << SPACE;
  os << csr.JA[csr.NZ - 1] << LINE_SEPARATOR;

  for(auto i = 0; i < csr.NZ - 1; i++)
    os << csr.AS[i] << SPACE;
  os << csr.AS[csr.NZ - 1] << LINE_SEPARATOR;

  return os;
}

std::istream & sspp::representations::operator >> (std::istream & is, sspp::representations::CSR & csr) {
  int *IRP, *JA;
  FLOATING_TYPE *AS;

  is >> csr.M;
  is >> csr.N;
  is >> csr.NZ;

  IRP = new int[csr.M + 1];
  JA = new int[csr.NZ];
  AS = new FLOATING_TYPE[csr.NZ];

  for(auto i = 0; i < csr.M + 1; i++)
    is >> IRP[i];
  for(auto i = 0; i < csr.NZ; i++)
    is >> JA[i];
  for(auto i = 0; i < csr.NZ; i++)
    is >> AS[i];

  if(csr.AS != 0)
    delete[] csr.AS;
  if(csr.IRP != 0)
    delete[] csr.IRP;
  if(csr.JA != 0)
    delete[] csr.JA;

  csr.AS = AS;
  csr.IRP = IRP;
  csr.JA = JA;

  return is;
}
