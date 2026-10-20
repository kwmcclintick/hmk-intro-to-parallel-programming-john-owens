#include <iostream>
#include <cuda_runtime.h>

__global__
void rgba_to_greyscale(const uchar4* rgbImage, unsigned char* const greyImage, int numRows, int numCols) {
    // TODO: .299f for r, .587f for g, .114f for b
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if( r >= numRows || c >= numCols ) return;

    int idx = r * numCols + c;
    greyImage[idx] = 0.299f * rgbImage[idx].x +
                      0.587f * rgbImage[idx].y +
                      0.114f * rgbImage[idx].z;
}

int main() {
    std::cout << "Start" << std::endl;
    // create Image
    int n = 128;
    int uc_bytes = n * n * sizeof(unsigned char);
    int uc4_bytes = n * n * sizeof(uchar4);
    uchar4 h_rgb_image[n][n];
    unsigned char h_grey_image[n][n];

    // fill image with some values
    for( int r = 0; r < n; ++r ) {
        for( int c = 0; c < n; ++c ) {
            h_rgb_image[r][c].x = r+1; // red
            h_rgb_image[r][c].y = c+1; // green
            h_rgb_image[r][c].z = r*c % 256; // blue
        }
    }

    // allocate gpu
    uchar4 *d_rgb_image; unsigned char* d_grey_image;
    cudaMalloc(&d_rgb_image, uc4_bytes);
    cudaMalloc(&d_grey_image, uc_bytes);

    // move image to gpu
    cudaMemcpy(d_rgb_image, h_rgb_image, uc4_bytes, cudaMemcpyHostToDevice);

    // launch
    const dim3 blockSize(16,16,1);
    const dim3 gridSize((n - blockSize.x - 1) / blockSize.x,
                        (n - blockSize.y - 1) / blockSize.y,1);
    std::cout << "launch" << std::endl;
    rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgb_image, d_grey_image, n, n);

    // move result back to cpu
    cudaMemcpy(h_grey_image, d_grey_image, uc_bytes, cudaMemcpyDeviceToHost);

    // print some results
    std::cout << "printing" << std::endl;
    for( int r = 5; r < 8; ++r ) {
        for( int c = 5; c < 8; ++c ) {
            std::cout << "r=" << r << ",c=" << c << ",value=" << static_cast<int>(h_grey_image[r][c] - '0') << std::endl;
        }
    }

    // free
    cudaFree(d_rgb_image);
    cudaFree(d_grey_image);

}
