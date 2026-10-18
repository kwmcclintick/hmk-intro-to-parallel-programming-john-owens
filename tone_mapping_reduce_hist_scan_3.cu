#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <chrono>

// ============================================================================
// CUDA KERNEL SIGNATURES (Empty for implementation)
// ============================================================================

struct minmax {
    float min{INFINITY};
    float max{-INFINITY};
    __device__ void compare(const minmax& mm ) {
        min = fminf(min, mm.min);
        max = fmaxf(max, mm.max);
    }
};

__global__
void find_min_max_kernel_stage_one(const float* __restrict__ d_logLuminance,
                                   minmax* __restrict__ d_stage_one_lum,
                                   const int numPixels) {
    // TODO: Step 1 - Find the minimum and maximum log-luminance of the image
    // Step 1.1.1 - write global memory into shared memory while doing thread level reduction
    __shared__ minmax shared[256];
    const int items_per_thread = 4;
    const int tid = threadIdx.x;
    const int gid = tid + (items_per_thread * blockIdx.x * blockDim.x); // each block processes 4 blocks worth of global memory
    minmax mm{};
    for( int i = 0; i < items_per_thread; ++i ) {
        const int offset = i*blockDim.x;
        if( gid + offset < numPixels ) mm.compare({d_logLuminance[gid+offset], d_logLuminance[gid+offset]});
    }
    shared[tid] = mm;
    __syncthreads();

    // step 1.1.2 - reduce, up until the last wrap
    const int threads_per_warp = 32;
    for( unsigned int s = blockDim.x / 2; s >= threads_per_warp; s >>= 1 ){
        if( tid < s ) shared[tid].compare(shared[tid + s]);
        __syncthreads();
    }

    // step 1.1.3 - reduce the last warp to avoid 5x syncs at 16, 8, 4, 2, 1
    if( tid < threads_per_warp ) {
        // 1.1.3.1 pull data from shm into local registers
        float my_min = shared[tid].min;
        float my_max = shared[tid].max;

        // 1.1.3.2 read from other threads registers in warp
        const unsigned int FULL_MASK = 0xFFFFFFFF;
        my_min = fminf( my_min, __shfl_down_sync(FULL_MASK, my_min, 16) );
        my_max = fmaxf( my_max, __shfl_down_sync(FULL_MASK, my_max, 16) );

        my_min = fminf( my_min, __shfl_down_sync(FULL_MASK, my_min, 8) );
        my_max = fmaxf( my_max, __shfl_down_sync(FULL_MASK, my_max, 8) );

        my_min = fminf( my_min, __shfl_down_sync(FULL_MASK, my_min, 4) );
        my_max = fmaxf( my_max, __shfl_down_sync(FULL_MASK, my_max, 4) );

        my_min = fminf( my_min, __shfl_down_sync(FULL_MASK, my_min, 2) );
        my_max = fmaxf( my_max, __shfl_down_sync(FULL_MASK, my_max, 2) );

        my_min = fminf( my_min, __shfl_down_sync(FULL_MASK, my_min, 1) );
        my_max = fmaxf( my_max, __shfl_down_sync(FULL_MASK, my_max, 1) );

        // step 1.1.1.4 write out answer
        if( tid == 0 ) d_stage_one_lum[blockIdx.x] = {my_min, my_max};
    }
}

__global__ void find_min_max_kernel_stage_two(const minmax* __restrict__ d_stage_one_lum, // size 300
                                              minmax* __restrict__ d_reduce_minmax, // size 1
                                              const int stage_one_blocks) {
    // Step 1.2.1 cache data into shm
    __shared__ minmax shared[256];
    const int tid = threadIdx.x;
    shared[tid] = d_stage_one_lum[tid];
    // we have less threads than data, so some need to double up
    if( tid + blockDim.x < stage_one_blocks ) { // given 256 and 300, tids < 44
        shared[tid].compare(d_stage_one_lum[tid+blockDim.x]);
    }
    __syncthreads();

    // Step 1.2.2 - reduce except for last wrap
    const int threads_per_warp = 32;
    for( unsigned int s = 128; s >= threads_per_warp; s >>= 1 ) {
        if( tid < s ) shared[tid].compare(shared[tid+s]);
        __syncthreads();
    }

    // Step 1.2.3 - reduce last warp
    if( tid < threads_per_warp ) {
        // write shm into registers
        float my_min = shared[tid].min;
        float my_max = shared[tid].max;

        // read registers of other threads in wrap_paragraphs
        const unsigned int FULL_MASK = 0xFFFFFFFF;
        my_min = fminf( my_min, __shfl_down_sync(FULL_MASK, my_min, 16) );
        my_max = fmaxf( my_max, __shfl_down_sync(FULL_MASK, my_max, 16) );

        my_min = fminf( my_min, __shfl_down_sync(FULL_MASK, my_min, 8) );
        my_max = fmaxf( my_max, __shfl_down_sync(FULL_MASK, my_max, 8) );

        my_min = fminf( my_min, __shfl_down_sync(FULL_MASK, my_min, 4) );
        my_max = fmaxf( my_max, __shfl_down_sync(FULL_MASK, my_max, 4) );

        my_min = fminf( my_min, __shfl_down_sync(FULL_MASK, my_min, 2) );
        my_max = fmaxf( my_max, __shfl_down_sync(FULL_MASK, my_max, 2) );

        my_min = fminf( my_min, __shfl_down_sync(FULL_MASK, my_min, 1) );
        my_max = fmaxf( my_max, __shfl_down_sync(FULL_MASK, my_max, 1) );

        // Step 1.2.4 - write output
        if( tid == 0 ) *d_reduce_minmax = {my_min, my_max};
    }
}

struct histogram {
    int bins[1024];
    __device__
    void operator+=(const histogram& hist) {
        for( int i = 0; i < 1024; ++i ) bins[i] += hist.bins[i];
    }
};

__global__ void histogram_kernel(const float* __restrict__ d_logLuminance,
                                 const minmax* __restrict__ d_reduce_minmax,
                                 const int numPixels, const int numBins,
                                 histogram* __restrict__ d_histogram) { // 16x smaller than it needs to be to have 1 histogram per lum data point
    // TODO: Step 2 - Generate a histogram of log-luminance values
    __shared__ histogram hist; // create in SRAM
    const int tid = threadIdx.x;
    // 256 threads per 1024 bins, so each thread will zero out four bins
    for( int i = tid; i < numBins; i += blockDim.x ) {
        hist.bins[i] = 0;
    }
    __syncthreads();
    // grab 16 values from global VRAM, compute their bin, and increment that bin
    // 16 values over 75 blocks over 256 threads gets us 640*480 pixels
    const int items_per_thread = 16;
    const int gid = (blockDim.x * blockIdx.x * items_per_thread) + tid;
    const float min_lum = d_reduce_minmax->min;
    const float max_lum = d_reduce_minmax->max;
    const float lum_range = max_lum - min_lum;
    for( int i = 0; i < items_per_thread; ++i ) {
        const int current_idx = gid + i*blockDim.x;
        if( current_idx < numPixels ) {
          const float lum = d_logLuminance[current_idx];
          const int hist_idx = fminf(static_cast<float>(numBins-1),
                                     ((lum - min_lum) / lum_range) * numBins );
          // 256 threads competing for 1024 bins, we need atomics to avoid races
          atomicAdd(&hist.bins[hist_idx],1);
        }
    }
    __syncthreads(); // make sure SRAM is full

    // write SRAM out to VRAM
    // as in the zeroing, we'll have each of the 256 threads write 4 of the 1024 bins to the one of 75 block histograms
    for( int i = tid; i < numBins; i += blockDim.x ) {
        atomicAdd(&d_histogram->bins[i], hist.bins[i]);
    }
}

__global__ void reduce_hist_kernel(const histogram* __restrict__ d_histograms,
                                   const int numHists,
                                   histogram* __restrict__ d_total_hist) {
    // combine 75 histograms into one, by each of the 1024 columns
    const int gid = blockDim.x * blockIdx.x + threadIdx.x; // bin Number
    for( int i = 0; i < numHists; ++i ) {
        (*d_total_hist).bins[gid] += d_histograms[i].bins[gid];
        // no one else will be rmw'ing the gid'th index of anything
        // so no need to sync
    }

}

__global__ void scan_kernel(histogram* __restrict__ d_total_hist,
                            const int numBins,
                            unsigned int* __restrict d_cdf) {
    // TODO: Step 3 - Compute the exclusive prefix sum (CDF) of the histogram
    // we're only going to read each hist value twice, but its better
    // to read once from VRAM and once from SRAM than twice from VRAM
    __shared__ unsigned int s_hist[1024];
    const int tid = threadIdx.x;
    s_hist[tid] = (*d_total_hist).bins[tid];
    __syncthreads();

    // Only the first step of hillis steele reads from this histogram
    __shared__ unsigned int s_cdf[1024];
    const int tid_hist_value = s_hist[tid]; // first read
    if( tid > 0 ) { // only first tid won't have a left neighbor
        const int neighbor_value = s_hist[tid-1]; // second read
        s_cdf[tid] = tid_hist_value + neighbor_value;
    } else {
        s_cdf[tid] = tid_hist_value;
    }
    __syncthreads();

    // hillis steele
    // add neighbor 1 left
    // add neighbor 2 left,
    // add neighbor 3 left. If no left neighbor, carry down
    for( unsigned int s = 2; s <= numBins/2; s <<= 1 ) {
        // start getting two values to the left, we already did the first step

        // at this point we're read-modify-writing the s_cdf, so we need to
        // have each thread read and compute their new value on registers
        // sync, then all write together, then sync again
        const int neighbor_idx = tid-s;
        unsigned int new_value = s_cdf[tid]; // if no left neighbor, carry down (aka, leave value as is)
        if( neighbor_idx >= 0 ) {
            new_value += s_cdf[neighbor_idx]; // if there's a neighbor, add it to what we'll set to
        }
        __syncthreads(); // once everyone has their value they'll set to, do all writes
        s_cdf[tid] = new_value;
        __syncthreads(); // wait for all writes before reading on next loop
    }

    // write out block SRAM to VRAM
    // each thread can write one Value,
    if( tid < 1024 ) d_cdf[tid] = s_cdf[tid] - s_hist[tid]; // subtract own value to make exclusive
}

// ============================================================================
// HOST REFERENCE FOR CORRECTNESS CHECK
// ============================================================================

void reference_tone_mapping(const float* const h_logLuminance,
                            unsigned int* const h_cdf,
                            const size_t numRows,
                            const size_t numCols,
                            const size_t numBins,
                            float &min_logLum,
                            float &max_logLum)
{
    size_t numPixels = numRows * numCols;

    // 1. Reference Min/Max
    min_logLum = h_logLuminance[0];
    max_logLum = h_logLuminance[0];
    for (size_t i = 1; i < numPixels; ++i) {
        min_logLum = std::min(min_logLum, h_logLuminance[i]);
        max_logLum = std::max(max_logLum, h_logLuminance[i]);
    }

    float logLumRange = max_logLum - min_logLum;

    // 2. Reference Histogram
    std::vector<unsigned int> h_histo(numBins, 0);
    for (size_t i = 0; i < numPixels; ++i) {
        unsigned int bin = std::min(
            static_cast<unsigned int>(numBins - 1),
            static_cast<unsigned int>((h_logLuminance[i] - min_logLum) / logLumRange * numBins)
        );
        h_histo[bin]++;
    }

    // 3. Reference Exclusive Scan (CDF)
    h_cdf[0] = 0;
    for (size_t i = 1; i < numBins; ++i) {
        h_cdf[i] = h_cdf[i - 1] + h_histo[i - 1];
    }
}

// ============================================================================
// MAIN PIPELINE
// ============================================================================

int main() {
    // Configurable parameters matching common course test cases
    const size_t numRows = 480;
    const size_t numCols = 640;
    const size_t numPixels = numRows * numCols;
    const size_t numBins = 1024;

    // Allocate Host Memory
    std::vector<float> h_logLuminance(numPixels);
    std::vector<unsigned int> gpu_cdf_output(numBins, 0);
    std::vector<unsigned int> ref_cdf_output(numBins, 0);

    // Initialize Host Input Data with synthetic logarithmic variations
    for (size_t i = 0; i < numPixels; ++i) {
        h_logLuminance[i] = log10f(1.0f + static_cast<float>(i % 255) / 10.0f);
    }

    // Allocate Device Memory
    // I/O
    float* d_logLuminance = nullptr;
    unsigned int* d_cdf = nullptr;

    // stage one part one
    cudaMalloc(&d_logLuminance, sizeof(float) * numPixels);
    cudaMalloc(&d_cdf, sizeof(unsigned int) * numBins);

    // stage one part two
    minmax* d_reduce_minmax = nullptr;
    cudaMalloc(&d_reduce_minmax, sizeof(minmax));

    // stage two
    const int threads_per_block = 256;
    int items_per_thread = 4;
    int blocks = (numPixels + threads_per_block - 1) / threads_per_block / items_per_thread; // 300 instead of 1,200 because of 4x per thread
    minmax* d_reduce_stage_one = nullptr;
    cudaMalloc(&d_reduce_stage_one, blocks * sizeof(minmax));

    items_per_thread = 16;
    blocks = (numPixels + threads_per_block - 1) / threads_per_block / items_per_thread; // 75 instead of 1,200 because of 16x per thread
    histogram* d_histogram = nullptr;
    const int total_hist_bytes = sizeof(histogram);
    cudaMalloc(&d_histogram, total_hist_bytes);
    cudaMemset(d_histogram, 0, sizeof(histogram));

    // Copy Input Data to Device
    cudaMemcpy(d_logLuminance, h_logLuminance.data(), sizeof(float) * numPixels, cudaMemcpyHostToDevice);
    auto gpu_start = std::chrono::steady_clock::now();

    // ------------------------------------------------------------------------
    // TODO: LAUNCH YOUR KERNELS HERE
    // ------------------------------------------------------------------------

    // Step 1: Find min/max values
    find_min_max_kernel_stage_one<<<300, 256>>>(
        d_logLuminance, d_reduce_stage_one, numPixels
    );
    find_min_max_kernel_stage_two<<<1, 256>>>(
        d_reduce_stage_one, d_reduce_minmax, blocks
    );

    // Step 2: Generate Histogram
    histogram_kernel<<<75, 256>>>(
        d_logLuminance, d_reduce_minmax,  numPixels, numBins, d_histogram
    );

    // Step 3: Prefix sum / Scan (CDF)
    // we only have 1024 items to scan, so we'd prefer
    // Hillis Steele's logn steps are preferred to Blelloch's 2*logn steps,
    // even if the work is n*logn compared to n
    scan_kernel<<<1,1024>>>(d_histogram, numBins, d_cdf);

    // ------------------------------------------------------------------------

    auto gpu_stop = std::chrono::steady_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_stop - gpu_start);
    std::cout << "GPU elapsed time: " << gpu_duration.count() << " microseconds\n";

    // Copy results back to host for assertion
    cudaMemcpy(gpu_cdf_output.data(), d_cdf, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost);

    // Compute serial reference standard
    float ref_min = 0.0f;
    float ref_max = 0.0f;
    auto start = std::chrono::steady_clock::now();
    reference_tone_mapping(h_logLuminance.data(), ref_cdf_output.data(), numRows, numCols, numBins, ref_min, ref_max);
    auto stop = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "CPU elapsed time: " << duration.count() << " microseconds\n";
    // ============================================================================
    // ASSERTION TESTS
    // ============================================================================
    std::cout << "Verifying CUDA implementation against reference..." << std::endl;

    for (size_t i = 0; i < numBins; ++i) {
        // Assert checks to verify matching outputs per bin index
        //std::cout << "Computed value=" << gpu_cdf_output[i] << ", expected value=" << ref_cdf_output[i] << std::endl;
        assert(gpu_cdf_output[i] == ref_cdf_output[i] && "Mismatch found between GPU and Reference CDF outputs!");
    }

    std::cout << "SUCCESS! Your parallel tone-mapping code is correct." << std::endl;

    // Cleanup resources
    cudaFree(d_logLuminance);
    cudaFree(d_cdf);

    return 0;
}
