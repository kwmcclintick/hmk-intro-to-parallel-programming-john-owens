#include <iostream>
#include <cuda_runtime.h>

#define IMAGE_WIDTH 128
#define BLOCK_SIZE 16

__global__ void blur(uchar4 *unblurred, uchar4* blurred, int rows, int cols) {
    // compute index and return if needed
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if( row >= rows || col >= cols) return;

    // since we're gathering (stencil), let's promote the global memory to shared memory
    int global_idx = col + row * cols; // row major
    int block_row = threadIdx.y;
    int block_col = threadIdx.x;
    __shared__ uchar4 cached_unblurred[BLOCK_SIZE][BLOCK_SIZE];
    cached_unblurred[block_row][block_col] = unblurred[global_idx];
    __syncthreads(); // wait for all data to be cached

    // once again row major
    // border items will have some loop divergence, worst case 5 loops waiting for corners
    int sumx = 0; int sumy = 0; int sumz = 0; int count = 0;
    for( int i = -1; i <= 1; ++i ) {
        for( int j = -1; j <= 1; ++j ) {
            int loop_row = row + i;
            int loop_col = col + j;
            // border check
            if( loop_row < 0 || loop_row >= rows || loop_col < 0 || loop_col >= cols ) continue;

            // read from shared if we can
            int bl_row = block_row + i;
            int bl_col = block_col + j;
            if( bl_row < blockDim.y && bl_col < blockDim.x && bl_row >= 0 && bl_col >= 0) {
                sumx += cached_unblurred[bl_row][bl_col].x;
                sumy += cached_unblurred[bl_row][bl_col].y;
                sumz += cached_unblurred[bl_row][bl_col].z;
            } else {  // around border of block
                int loop_idx = loop_col + loop_row * cols;
                sumx += unblurred[loop_idx].x;
                sumy += unblurred[loop_idx].y;
                sumz += unblurred[loop_idx].z;
            }
            count++;
        }
    }

    // perform writes
    blurred[global_idx].x = sumx / count;
    blurred[global_idx].y = sumy / count;
    blurred[global_idx].z = sumz / count;
}

int main() {
    // create image
    int n = IMAGE_WIDTH;
    int n_bytes = n * n * sizeof(uchar4);
    uchar4* h_unblurred = (uchar4*)malloc(n_bytes);
    for( int r = 0; r < n; ++r ) {
        for( int c = 0; c < n; ++c ) {
            int idx = r * n + c;
            h_unblurred[idx].x = r;
            h_unblurred[idx].y = c;
            h_unblurred[idx].z = 1;
        }
    }

    // allocate device i/o
    uchar4* d_blurred; uchar4* d_unblurred;
    cudaMalloc(&d_blurred, n_bytes);
    cudaMalloc(&d_unblurred, n_bytes);
    cudaMemset(d_blurred, 0, n_bytes);

    // memcpy to device
    cudaMemcpy(d_unblurred, h_unblurred, n_bytes, cudaMemcpyHostToDevice);

    // launch kernel
    const dim3 blockDim(BLOCK_SIZE,BLOCK_SIZE,1); // 256 threads / block
    const dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                       (n + blockDim.y - 1) / blockDim.y, 1);
    int shared_bytes = blockDim.x * blockDim.y * sizeof(uchar4);
    blur<<<gridDim, blockDim, shared_bytes>>>(d_unblurred, d_blurred, n, n);
    cudaDeviceSynchronize();

    // memcpy back to hooks
    uchar4* h_blurred = (uchar4 *)malloc(n_bytes);
    cudaMemcpy(h_blurred, d_blurred, n_bytes, cudaMemcpyDeviceToHost);

    // print
    for( int r = 0; r < 3; ++r ) {
        for( int c = 0; c < 3; ++c ) {
            int idx = r * n + c;
            std::cout << "r=" <<r<<",c="<<c<<",val="<<static_cast<int>(h_blurred[idx].x)<< std::endl;
        }
    }

    // free
    cudaFree(d_blurred); cudaFree(d_unblurred);
    free(h_blurred); free(h_unblurred);
}
