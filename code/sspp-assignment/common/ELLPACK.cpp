#include "ELLPACK.h"

void sspp::representations::ELLPACK::rewrite(ELLPACK & lhs, const ELLPACK & rhs) {
  lhs.M = rhs.M;
  lhs.N = rhs.N;
  lhs.NZ = rhs.NZ;
  lhs.MAXNZ = rhs.MAXNZ;

  lhs.AS = new FLOATING_TYPE*[rhs.M];
  for(auto i = 0; i < rhs.M; i++) {
    lhs.AS[i] = new FLOATING_TYPE[rhs.MAXNZ];
    for(auto j = 0; j < rhs.MAXNZ; j++)
      lhs.AS[i][j] = rhs.AS[i][j];
  }

  lhs.JA = new int*[rhs.M];
  for(auto i = 0; i < rhs.M; i++) {
    lhs.JA[i] = new int[rhs.MAXNZ];
    for(auto j = 0; j < rhs.MAXNZ; j++)
      lhs.JA[i][j] = rhs.JA[i][j];
  }
}

sspp::representations::ELLPACK::ELLPACK()
  : M(0), N(0), NZ(0), MAXNZ(0), JA(0), AS(0) {

}

sspp::representations::ELLPACK::ELLPACK(int M, int N, int NZ, int MAXNZ, int ** JA, FLOATING_TYPE ** AS)
  : M(M), N(N), NZ(NZ), MAXNZ(MAXNZ), JA(JA), AS(AS) {
}

sspp::representations::ELLPACK::ELLPACK(const ELLPACK & other) {
  rewrite(*this, other);
}

sspp::representations::ELLPACK & sspp::representations::ELLPACK::operator=(representations::ELLPACK rhs) {
  rewrite(*this, rhs);
  return *this;
}

sspp::representations::ELLPACK::~ELLPACK() {
  for(auto i = 0; i < M; i++)
    delete[] JA[i];
  delete[] JA;

  for(auto i = 0; i < M; i++)
    delete[] AS[i];
  delete[] AS;
}

std::ostream & sspp::representations::operator<<(std::ostream & os, const ELLPACK & ellpack) {
  os << ellpack.M << LINE_SEPARATOR;
  os << ellpack.N << LINE_SEPARATOR;
  os << ellpack.NZ << LINE_SEPARATOR;
  os << ellpack.MAXNZ << LINE_SEPARATOR;

  for(auto i = 0; i < ellpack.M; i++) {
    for(auto j = 0; j < ellpack.MAXNZ - 1; j++) {
      os << ellpack.JA[i][j] << SPACE;
    }
    os << ellpack.JA[i][ellpack.MAXNZ - 1] << LINE_SEPARATOR;
  }

  for(auto i = 0; i < ellpack.M; i++) {
    for(auto j = 0; j < ellpack.MAXNZ - 1; j++) {
      os << ellpack.AS[i][j] << SPACE;
    }
    os << ellpack.AS[i][ellpack.MAXNZ - 1] << LINE_SEPARATOR;
  }

  return os;
}

std::istream & sspp::representations::operator >> (std::istream & is, ELLPACK & ellpack) {
  int **JA;
  FLOATING_TYPE **AS;

  is >> ellpack.M;
  is >> ellpack.N;
  is >> ellpack.NZ;
  is >> ellpack.MAXNZ;

  JA = new int*[ellpack.M];
  for(auto i = 0; i < ellpack.M; i++)
    JA[i] = new int[ellpack.MAXNZ];

  AS = new FLOATING_TYPE*[ellpack.M];
  for(auto i = 0; i < ellpack.M; i++)
    AS[i] = new FLOATING_TYPE[ellpack.MAXNZ];

  for(auto i = 0; i < ellpack.M; i++) {
    for(auto j = 0; j < ellpack.MAXNZ; j++) {
      is >> JA[i][j];
    }
  }

  for(auto i = 0; i < ellpack.M; i++) {
    for(auto j = 0; j < ellpack.MAXNZ; j++) {
      is >> AS[i][j];
    }
  }

  if(ellpack.JA != 0) {
    for(auto i = 0; i < ellpack.M; i++)
      delete[] ellpack.JA[i];
    delete[] ellpack.JA;
  }

  if(ellpack.AS != 0) {
    for(auto i = 0; i < ellpack.M; i++)
      delete[] ellpack.AS[i];
    delete[] ellpack.AS;
  }

  ellpack.JA = JA;
  ellpack.AS = AS;

  return is;
}
