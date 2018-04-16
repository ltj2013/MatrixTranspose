
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include <stdio.h>
#include <time.h>
#include <iostream>


#include "Matrix.h"
#include "config.h"
#include "SimpleIm.h"
#include "SharedMatrixTranspose.h"
#include "Func.cu"

using namespace std;


//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
	const int WIDTH = 4096 ;
	const int HEIGHT = 4096;
	const int Length = WIDTH*HEIGHT;
	//int array[1048576];
	//Matrix<int, WIDTH, HEIGHT> matrix(array);
	
	float *input = new float[Length];
	for (int i = 0; i < Length; i++)
		input[i] = i;
	const int num_rows = WIDTH;
	const int num_cols = HEIGHT;

	float *input_matrix;
	float *output_matrix;

	cudaMalloc((void**)&input_matrix, sizeof(float) * HEIGHT *HEIGHT);
	cudaMalloc((void**)&output_matrix, sizeof(float) * HEIGHT *HEIGHT);

	cudaMemcpy(input_matrix, input, sizeof(float) * HEIGHT *HEIGHT, cudaMemcpyHostToDevice);

	int grid_size = (num_rows - 1) / BLOCKSIZE + 1;
	dim3 gridDim(grid_size, grid_size);
	dim3 blockDim(BLOCKSIZE, BLOCKSIZE);

	clock_t time = clock();

	simpleIMP <<<gridDim, blockDim >>> (input_matrix, output_matrix, num_rows, num_cols);

	clock_t result = clock() - time;
	sharedMatrixTranspose << <gridDim, blockDim >> > (input_matrix, output_matrix, num_rows, num_cols);

	clock_t result2 = clock() - result;

	cudaMemcpy(input,output_matrix, sizeof(float) * HEIGHT * HEIGHT, cudaMemcpyDeviceToHost);


	//record the execution time 
	float gpu_elapsed_time_ms = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	cudaEventRecord(start, 0);
	baseTranspose<float> << <gridDim, blockDim >> > (input_matrix, output_matrix, num_rows, num_cols);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
	
	std::cout << gpu_elapsed_time_ms << std::endl;

	cudaEventRecord(start, 0);

	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	coalescedBlockWiseTranspose<float> << <gridDim, blockDim,0,stream1>> > (input_matrix, output_matrix, num_rows, num_cols);
	coalescedBlockWiseTransposeWithNoBankConflicts<float> << <gridDim, blockDim,0,stream2>> > (input_matrix, output_matrix, num_rows, num_cols);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);


	
	std::cout << gpu_elapsed_time_ms << std::endl;

	cudaFree(input_matrix);
	cudaFree(output_matrix);
//	for (int i = 0; i < 1024; i++)
		//cout<<input[i]<<endl;
	return 0;
}
