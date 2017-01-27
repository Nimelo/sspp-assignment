// 
// Author: Salvatore Filippone salvatore.filippone@cranfield.ac.uk
//

// Computes matrix-vector product. Matrix A is in row-major order
// i.e. A[i, j] is stored in i * COLS + j element of the vector.
//

#include <iostream>

#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers
#include <ctime>
#include <algorithm>

// Matrix dimensions.
const int ROWS = 4096;
const int COLS = 4096;

// TODO(later) Play a bit with the block size. Is 16x16 setup the fastest possible?
// Note: For meaningful time measurements you need sufficiently large matrix.
const int BLOCK_DIM_X = 4;
const int BLOCK_DIM_Y = 4;
const dim3 BLOCK_DIM(BLOCK_DIM_X, BLOCK_DIM_Y);

// Simple CPU implementation of matrix addition.
void CpuMatrixVector(int rows, int cols, const double* A, const double* x, double* y) {
	for (int row = 0; row < rows; ++row) {
		double t = 0.0;
		for (int col = 0; col < cols; ++col) {
			int idx = row * cols + col;
			t += A[idx] * x[col];
		}
		y[row] = t;
	}
}

// GPU implementation of matrix add using one CUDA thread per vector element.
__global__ void gpuMatrixVector(int rows, int cols, const double* A, const double* x, double* y) {
	// TODO Calculate indices of matrix elements added by this thread. 
	//int idx = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);
	//int col = 0;
	int idx = threadIdx.x + threadIdx.y * blockDim.x + blockDim.x * blockDim.y * blockIdx.x;
	double t = 0.0;
	for (int i = 0; i<cols; i++)
	{
		t += A[rows * idx + i] * x[i];
	}
	y[idx] = t;

	// TODO Calculate the element index in the global memory and add the values.
	// TODO Make sure that no threads access memory outside the allocated area.
}

/*
// GPU implementation of matrix add using one CUDA thread per matrix element.
__global__ void gpuMatrixVector1(int rows, int cols, const double* A, const double* x, double* y) {
// TODO Calculate indices of matrix elements added by this thread.
int idx = threadIdx.x + blockDim.x * ( blockIdx.x + blockIdx.y * gridDim.x );
int col = idx%cols;
//int row = idx/cols;
if(idx < rows*cols)
{
y[col] += A[idx] * x[col];
}

// TODO Calculate the element index in the global memory and add the values.
// TODO Make sure that no threads access memory outside the allocated area.
}
*/

int main(int argc, char** argv) {

	// ----------------------- Host memory initialisation ----------------------- //

	double* h_A = new double[ROWS * COLS];
	double* h_x = new double[COLS];
	double* h_y = new double[ROWS];
	double* h_y_d = new double[ROWS];

	srand(time(0));
	for (int row = 0; row < ROWS; ++row) {
		for (int col = 0; col < COLS; ++col) {
			int idx = row * COLS + col;
			h_A[idx] = 100.0f * static_cast<double>(rand()) / RAND_MAX;
		}
		h_x[row] = 100.0f * static_cast<double>(rand()) / RAND_MAX;
	}

	// ---------------------- Device memory initialisation ---------------------- //

	// [TODO-OK] Allocate global memory on the GPU.
	double *d_A, *d_x, *d_y;
	cudaMalloc((void**)&d_A, ROWS*COLS * sizeof(double));
	cudaMalloc((void**)&d_x, COLS * sizeof(double));
	cudaMalloc((void**)&d_y, ROWS * sizeof(double));

	// [TODO-OK] Copy matrices from the host (CPU) to the device (GPU).
	cudaMemcpy(d_A, h_A, ROWS*COLS * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, h_x, COLS * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, ROWS * sizeof(double), cudaMemcpyHostToDevice);

	// ------------------------ Calculations on the CPU ------------------------- //

	// Create the CUDA SDK timer.
	StopWatchInterface* timer = 0;
	sdkCreateTimer(&timer);

	timer->start();
	CpuMatrixVector(ROWS, COLS, h_A, h_x, h_y);

	timer->stop();
	std::cout << "CPU time: " << timer->getTime() << " ms." << std::endl;

	// ------------------------ Calculations on the GPU ------------------------- //

	// [TODO-OK] Calculate the dimension of the grid of blocks (2D).
	//const dim3 GRID_DIM;
	////const dim3 GRID_DIM = (ROWS*COLS+BLOCK_DIM_X*BLOCK_DIM_Y-1)/(BLOCK_DIM_X*BLOCK_DIM_Y);	//one thread per matrix value.  = 65536
	dim3 GRID_DIM = ROWS / (BLOCK_DIM.x * BLOCK_DIM.y);

	timer->reset();
	timer->start();
	gpuMatrixVector << <GRID_DIM, BLOCK_DIM>> >(ROWS, COLS, d_A, d_x, d_y);
	checkCudaErrors(cudaDeviceSynchronize());

	timer->stop();
	std::cout << "GPU time: " << timer->getTime() << " ms." << std::endl;

	// [TODO-OK] Download the resulting vector d_y from the device and store it in h_y_d.
	cudaMemcpy(h_y_d, d_y, ROWS * sizeof(double), cudaMemcpyDeviceToHost);

	// Now let's check if the results are the same.
	double diff = 0.0f;
	for (int row = 0; row < ROWS; ++row) {
		diff = std::max(diff, std::abs(h_y[row] - h_y_d[row]));
	}
	std::cout << "Max diff = " << diff << std::endl;  // Should be (very close to) zero.

													  // ------------------------------- Cleaning up ------------------------------ //

	delete timer;

	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_x));
	checkCudaErrors(cudaFree(d_y));

	delete[] h_A;
	delete[] h_x;
	delete[] h_y;
	delete[] h_y_d;
	return 0;
}
