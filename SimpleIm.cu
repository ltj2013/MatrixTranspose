#include "SimpleIm.h"

__global__ void simpleIMP(float *input_matrx, float *output_matrix, const int num_cols, const int num_rows) {

	int thread_y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_x = blockIdx.x * blockDim.x + threadIdx.x;

	int index_x = thread_y * num_cols + thread_x;

	int index_output = thread_x*num_rows + thread_y;

	if (thread_y < num_rows && thread_x < num_cols) {
		output_matrix[index_output] = input_matrx[index_x];
	}
}