#include "CSRCudaSolver.h"
#include <host_defines.h>
#include <device_launch_parameters.h>

__global__ void csrKernel(representations::csr::CSR & csr, FLOATING_TYPE* b, representations::output::Output & output)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	if(row < csr.M)
	{
		FLOATING_TYPE dot = 0;
		int rowStart = csr.IRP[row],
			rowEnd = csr.IRP[row + 1];

		for (int i = rowStart; i < rowEnd; i++)
		{
			dot += csr.AS[i] * b[csr.JA[i]];
		}

		output.Values[row] += dot;
	}

}

representations::output::Output tools::solvers::csr::CSRCudaSolver::solve(representations::csr::CSR & csr, FLOATING_TYPE * b)
{
	FLOATING_TYPE *x = new FLOATING_TYPE[csr.M];

	return representations::output::Output(csr.M, x);
}
