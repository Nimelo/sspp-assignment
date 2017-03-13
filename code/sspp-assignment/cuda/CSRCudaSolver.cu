#include "CSRCudaSolver.h"
#include "ELLPACKCudaSolver.h"
#include <host_defines.h>

__global__ void csrKernel(representations::ellpack::ELLPACK & ellpack, FLOATING_TYPE* b, representations::output::Output & output)
{
	//TODO: Add csrKernel	
}

representations::output::Output tools::solvers::cuda::csr::CSRCudaSolver::solve(representations::csr::CSR & csr, FLOATING_TYPE * b)
{
	FLOATING_TYPE *x = new FLOATING_TYPE[csr.M];

	return representations::output::Output(csr.M, x);
}
