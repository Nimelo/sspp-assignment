//#ifndef __H_CSR_INVOKER
//#define __H_CSR_INVOKER
//
//#include <string>
//#include "..\common\Result.h"
//#include "..\common/AbstractCRSSolver.h"
//
//namespace sspp {
//  namespace tools {
//    namespace invokers {
//      class CSRInvoker {
//      protected:
//        std::string inputFile;
//        std::string outputFile;
//        int iterationsParallel;
//        int iterationsSerial;
//
//        common::CRS loadCSR();
//        std::vector<FLOATING_TYPE> createVectorB(int n);
//        void saveResult(representations::result::Result & result);
//      public:
//        CSRInvoker(std::string inputFile, std::string outputFile, int iterationsParallel, int iterationsSerial);
//        void invoke(solvers::AbstractCSRSolver & solver);
//      };
//    }
//  }
//}
//#endif
