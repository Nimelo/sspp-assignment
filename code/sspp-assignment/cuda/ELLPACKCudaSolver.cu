#include "ELLPACKCudaSolver.h"
#include <host_defines.h>

__global__ void ellpackKernel(representations::ellpack::ELLPACK & ellpack, FLOATING_TYPE* b, representations::output::Output & output)
{
	//TODO: Add ellpackKernel	
}

representations::output::Output tools::solvers::ellpack::ELLPACKCudaSolver::solve(representations::ellpack::ELLPACK & ellpack, FLOATING_TYPE * b)
{
	FLOATING_TYPE *x = new FLOATING_TYPE[ellpack.M];

	return representations::output::Output(ellpack.M, x);
}
