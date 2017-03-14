#include "ELLPACKCudaSolver.h"
#include <host_defines.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

__global__ void ellpackKernel(representations::ellpack::ELLPACK & ellpack, FLOATING_TYPE* b, FLOATING_TYPE *x)
{
	//TODO: Add ellpackKernel	
	int row = blockDim.x * blockIdx.x + threadIdx.x;

	if (row < ellpack.M)
	{
		FLOATING_TYPE dot = 0;
		for (int i = 0; i < ellpack.MAXNZ; i++)
		{
			int col = ellpack.JA[row][i];
			FLOATING_TYPE val = ellpack.AS[row][i];
			if (val != 0)
				dot += val * b[col];
		}
		x[row] += dot;
	}

}

representations::output::Output tools::solvers::ellpack::ELLPACKCudaSolver::solve(representations::ellpack::ELLPACK & ellpack, FLOATING_TYPE * b)
{
	using namespace representations::ellpack;
	FLOATING_TYPE *x = new FLOATING_TYPE[ellpack.M];
	cudaError_t cudaStatus;
	ELLPACK *hEllpackPtr = &ellpack;
	ELLPACK *dEllpackPtr;
	FLOATING_TYPE** h_AS = hEllpackPtr->AS;
	int** h_JA = hEllpackPtr->JA;

	int *d_JA;
	FLOATING_TYPE *d_AS, *d_B, *d_X;

	cudaMalloc(&d_JA, sizeof(int) * ellpack.M * ellpack.MAXNZ);
	cudaMemcpy(d_JA, h_JA, sizeof(int) * ellpack.M * ellpack.MAXNZ, cudaMemcpyHostToDevice);

	hEllpackPtr->JA = &d_JA;

	cudaMalloc(&d_AS, sizeof(FLOATING_TYPE) * ellpack.M * ellpack.MAXNZ);
	cudaMemcpy(d_AS, h_AS, sizeof(FLOATING_TYPE) * ellpack.M * ellpack.MAXNZ, cudaMemcpyHostToDevice);

	hEllpackPtr->AS = &d_AS;

	cudaMalloc(&d_B, sizeof(FLOATING_TYPE) * ellpack.M);
	cudaMemcpy(d_B, b, sizeof(FLOATING_TYPE) * ellpack.M * ellpack.MAXNZ, cudaMemcpyHostToDevice);

	cudaMalloc(&d_X, sizeof(FLOATING_TYPE) * ellpack.M);

	ellpackKernel << <ellpack.M, 1 >> > (ellpack, d_B, d_X);

	cudaMemcpy(x, d_X, sizeof(FLOATING_TYPE)*ellpack.M, cudaMemcpyDeviceToHost);

	return representations::output::Output(ellpack.M, x);
}
