#ifndef SSPP_COMMON_MATRIXMARKETSTREAM_H_
#define SSPP_COMMON_MATRIXMARKETSTREAM_H_

#include "MatrixMarketTuple.h"
#include "MatrixMarketHeader.h"
#include "MatrixMarketTupleReaderInterface.h"

#include <istream>
#include <sstream>

namespace sspp {
  namespace common {
    template<typename VALUE_TYPE>
    class MatrixMarketStream {
    public:
      MatrixMarketStream(std::istream & stream,
                         MatrixMarketTupleReaderInterface<VALUE_TYPE> &tuple_reader) :
        tuple_reader_(tuple_reader),
        stream_(stream) {
        stream_ >> header_;
        GoToEntries();
      };

      MatrixMarketTuple<VALUE_TYPE> GetNextTuple() {
        return tuple_reader_.Get(current_line_, header_.IsPattern());
      }

      bool hasNextTuple() {
        current_line_.clear();
        std::getline(stream_, current_line_);
        return !current_line_.empty();
      }

      MatrixMarketHeader const & GetMatrixMarketHeader() const {
        return header_;
      }

      unsigned GetRows() const {
        return rows_;
      }

      unsigned GetColumns() const {
        return columns_;
      }

      unsigned GetNonZeros() const {
        return non_zeros_;
      }

      void GoToEntries() {
        const char COMMENT_PREFIX_STR[] = "%";
        stream_.clear();
        stream_.seekg(0, stream_.beg);

        std::string line;
        while(!stream_.eof()) {
          std::getline(stream_, line);
          if(line.find_first_of(COMMENT_PREFIX_STR) != 0) {
            std::stringstream ss(line);
            ss >> rows_ >> columns_ >> non_zeros_;
            break;
          }
        }
      }

    private:
      unsigned rows_;
      unsigned columns_;
      unsigned non_zeros_;

      std::string current_line_;
      MatrixMarketTupleReaderInterface<VALUE_TYPE> &tuple_reader_;
      std::istream &stream_;
      MatrixMarketHeader header_;
    };
  }
}

#endif
