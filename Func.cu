#pragma once

#include <device_launch_parameters.h>

#include "config.h"

template <typename T>
__global__ void baseLineCopy(const T * input, T* output, int width, int height) {
	
	const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	const int index_x = blockIdx.x * blockDim.x + threadIdx.x;

	const int index = index_y * width + index_x;

	if (index_x < width && index_y < height) {
		output[index] = input[index];
	}
}

template<typename T>
__global__ void baseSharedCopy(const T *input, T *output, int width, int height) {

	__shared__ T temporary[BLOCKSIZE][BLOCKSIZE];

	const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	const int index_x = blockIdx.x * blockDim.x + threadIdx.x;

	const int index = index_y * width + index_x;

	if (index_y < height && index_x < width) {

		temporary[threadIdx.y][threadIdx.x] = input[index];
		__syncthreads();

		output[index] = temporary[threadIdx.y][threadIdx.x];
	}
}



template<typename T>
__global__ void baseTranspose(const T* input, T* output, int width, int height) {
	
	const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	const int index_x = blockIdx.x * blockDim.x + threadIdx.x;

	const int in_index = index_y * width + index_x;

	const int out_index = index_x * height + index_y;
	
	if ((index_x < width) && (index_y < height) ) {
		output[out_index] = input[in_index];
	}
}

template<typename T>
__global__ void baseBlockWiseTranspose(const T* input, T* output, int width, int height) {

	__shared__ T temporary[BLOCKSIZE][BLOCKSIZE];

	const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	const int index_x = blockIdx.x * blockDim.x + threadIdx.x;

	const int in_index = index_y * width + index_x;

	const int out_index = index_x * height + index_y;

	if (index_x < width && index_y < height) {
		temporary[threadIdx.x][threadIdx.y] = input[in_index];
		
		__syncthreads();

		output[out_index] = temporary[threadIdx.y][threadIdx.x];
	}
}


template<typename T>
__global__ void coalescedBlockWiseTranspose(const T* input, T* output, int width, int height) {

	__shared__ T temporary[BLOCKSIZE][BLOCKSIZE];

	const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	const int index_x = blockIdx.x * blockDim.x + threadIdx.x;

	const int in_index = index_y * width + index_x;

	int xIndexOut = blockIdx.y * blockDim.y + threadIdx.x;
	int yIndexOut = blockIdx.x * blockDim.x + threadIdx.y;
	int out_index = yIndexOut * height + xIndexOut;

	//swap the x and y in shared memory to implement memory coalescing 

	if (index_x < width && index_y < height) {
		temporary[threadIdx.y][threadIdx.x] = input[in_index];
		
		__syncthreads();
		//temporary[threadIdx.x][threadIdx.y] = temporary[threadIdx.y][threadIdx.x];
		//__syncthreads();
		output[out_index] = temporary[threadIdx.x][threadIdx.y];
		output[out_index] = temporary[threadIdx.x][threadIdx.y];
		output[out_index] = temporary[threadIdx.x][threadIdx.y];
		output[out_index] = temporary[threadIdx.x][threadIdx.y];
		output[out_index] = temporary[threadIdx.x][threadIdx.y];
		output[out_index] = temporary[threadIdx.x][threadIdx.y];
		output[out_index] = temporary[threadIdx.x][threadIdx.y];
	}
}

template<typename T>
__global__ void coalescedBlockWiseTransposeWithNoBankConflicts(const T* input, T* output, int width, int height) {

	__shared__ T temporary[BLOCKSIZE][BLOCKSIZE+1];

	const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	const int index_x = blockIdx.x * blockDim.x + threadIdx.x;

	const int in_index = index_y * width + index_x;

	int xIndexOut = blockIdx.y * blockDim.y + threadIdx.x;
	int yIndexOut = blockIdx.x * blockDim.x + threadIdx.y;
	int out_index = yIndexOut * height + xIndexOut; //输出地址是连续的

	//swap the x and y in shared memory to implement memory coalescing 

	if (index_x < width && index_y < height) {
		temporary[threadIdx.y][threadIdx.x] = input[in_index];

		__syncthreads();

		output[out_index] = temporary[threadIdx.x][threadIdx.y];
		output[out_index] = temporary[threadIdx.x][threadIdx.y];
		output[out_index] = temporary[threadIdx.x][threadIdx.y];
		output[out_index] = temporary[threadIdx.x][threadIdx.y];
		output[out_index] = temporary[threadIdx.x][threadIdx.y];
		output[out_index] = temporary[threadIdx.x][threadIdx.y];
		output[out_index] = temporary[threadIdx.x][threadIdx.y];
	}
}