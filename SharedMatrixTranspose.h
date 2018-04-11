#pragma once

__global__ void sharedMatrixTranspose(float *input_matrix, float * output_matrix, const int num_rows, const int num_cols);