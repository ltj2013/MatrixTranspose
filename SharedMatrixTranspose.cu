#include "SharedMatrixTranspose.h"
#include "config.h"

__global__ void sharedMatrixTranspose(float *input_matrix, float * output_matrix, const int num_rows, const int num_cols) {
	
	int thread_y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_x = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float matrix_data[BLOCKSIZE][BLOCKSIZE];

	int index_x = thread_y * num_cols + thread_x;
	int index_y = thread_x * num_rows + thread_y;

	if(thread_y < num_rows && thread_x < num_cols)
		matrix_data[threadIdx.x][threadIdx.y] = input_matrix[index_x];
	__syncthreads();

	if (thread_y < num_rows && thread_x < num_cols)
		output_matrix[index_y] = matrix_data[threadIdx.y][threadIdx.x];
	__syncthreads();
 }