#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Macro for simple CUDA error checking
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result) << "(\"" << cudaGetErrorString(result) << "\") \"" << func << "\" \n";
        cudaDeviceReset();
        exit(99);
    }
}

// Struct matching the reference_calc pattern for validation
struct SortPair {
    unsigned int val;
    unsigned int pos;
};

bool compareSortPairs(const SortPair& a, const SortPair& b) {
    return a.val < b.val;
}

// CPU reference calculator to verify your GPU code's accuracy
void reference_calculation(unsigned int* const inputVals,
                           unsigned int* const inputPos,
                           unsigned int* const outputVals,
                           unsigned int* const outputPos,
                           const size_t numElems)
{
    std::vector<SortPair> pairs(numElems);
    for (size_t i = 0; i < numElems; ++i) {
        pairs[i].val = inputVals[i];
        pairs[i].pos = inputPos[i];
    }

    std::stable_sort(pairs.begin(), pairs.end(), compareSortPairs);

    for (size_t i = 0; i < numElems; ++i) {
        outputVals[i] = pairs[i].val;
        outputPos[i]  = pairs[i].pos;
    }
}

// ============================================================================
// STUDENT IMPLEMENTATION AREA
// ============================================================================

__global__ void histogram_kernel(const unsigned int* __restrict__ d_inputVals,
                                 const size_t numElems,
                                 const size_t base_bit,
                                 const size_t radix_width,
                                 const size_t coarsening_factor,
                                 const size_t tpb,
                                 const size_t bins_per_block,
                                 unsigned int* __restrict__ d_histograms) {

    __shared__ unsigned int bins[16]; //
    const int tid = threadIdx.x; // 256 threads to zero out just 16 bins
    if( tid < bins_per_block ) {
      bins[tid] = 0;
    }
    __syncthreads();


    const unsigned int mask = (1 << radix_width) - 1; // 15=1111
    const int gid = (blockDim.x * coarsening_factor) * blockIdx.x + threadIdx.x;
    for( int i = 0; i < coarsening_factor; ++i ) {
      const int coarse_gid = gid + i*tpb;
      if( coarse_gid < numElems ) {
        const unsigned int val = d_inputVals[coarse_gid];
        const unsigned int bit_bin = ( val >> base_bit ) & mask;
        atomicAdd(&bins[bit_bin],1); // 256 threads fighting for 16 bins (1/16 chance of bank collision)
      }
    }
    __syncthreads();

    // 256 threads writing out just 8 bins
    if( tid < bins_per_block ) {
      d_histograms[tid * gridDim.x + blockIdx.x] = bins[tid];
      // block 0 digit 0, block 1 digit 0, block 2 digit 0,...
    }
}


__global__ void ex_scan_kernel(const unsigned int* __restrict__ d_histograms,
                               const size_t histogram_length,
                               const size_t coarsening_factor,
                               unsigned int* __restrict__ d_scan_output) {
  // 512 threads to scan 1568 items, most scan 3, some scan 4
  // First, each thread writes 3 (or 4) items from VRAM into SRAM
  const int tid = threadIdx.x;
  __shared__ unsigned int s_mem[2048]; // 1568 values with padding after
  for( int i = 0; i < coarsening_factor; ++i ){
    const int coarse_idx = tid + i * blockDim.x;
    if( coarse_idx < histogram_length ) {
      s_mem[coarse_idx] = d_histograms[coarse_idx]; // one block
      // 0,512,1024,1536
      // 1,513,1025,1537,....etc. most of these process just 3, not 4
    }
  }
  __syncthreads();

  // second, we do the first step in sram
  __shared__ unsigned int s_scan[2048]; // 1568 values, with padding after
  for( int i = 0; i < coarsening_factor; ++i ) {
    const int coarse_tid = tid + i*blockDim.x;
    if( coarse_tid < histogram_length ) {
      const unsigned int tid_value = s_mem[coarse_tid]; // first read
      if( coarse_tid > 0 ) {
        const unsigned int neighbor_value = s_mem[coarse_tid-1]; // second read
        s_scan[coarse_tid] = tid_value + neighbor_value; // write is to new array so no sync needed
      } else {
        s_scan[coarse_tid] = tid_value;
      }
    }
  }
  __syncthreads();

  // third, we finish the hillis steele scan
  for( unsigned int s = 2; s <= 1024; s <<= 1 ) {

    // first, each thread decides what their new value will be (reads)
    unsigned int vals[4]; // coarsening factor
    for( int i = 0; i < coarsening_factor; ++i ) {
      const int coarse_tid = tid + i*blockDim.x;
      if( coarse_tid < histogram_length ) {
        const int neighbor_idx = coarse_tid-s;
        vals[i] = s_scan[coarse_tid];
        if( neighbor_idx >= 0 ) {
          // if neighbor exists, add it
          vals[i] += s_scan[neighbor_idx];
        }
      }
    }
    __syncthreads();

    // then we do the writes all at once
    for( int i = 0; i < coarsening_factor; ++i ) {
      const int coarse_tid = tid + i*blockDim.x;
      if( coarse_tid < histogram_length ) {
        s_scan[coarse_tid] = vals[i];
      }
    }
    __syncthreads();
  }

  // Then, we write answer out to VRAM, subtracting the original value to make it exclusive
  for( int i = 0; i < coarsening_factor; ++i ) {
    const int coarse_tid = tid + i*blockDim.x;
    if( coarse_tid < histogram_length) {
      d_scan_output[coarse_tid] = s_scan[coarse_tid] - s_mem[coarse_tid];
    }
  }
}


__global__ void scatter_kernel(const unsigned int* __restrict__ inVals,
                               const unsigned int* __restrict__ inPos,
                               unsigned int* __restrict__ outVals,
                               unsigned int* __restrict__ outPos,
                               const unsigned int* __restrict__ scan,
                               const size_t numElems,
                               const size_t base_bit,
                               const size_t radix_width,
                               const size_t tile)          // tpb * coarsening = 1024
{
  __shared__ unsigned int s_digit[1024]; // tile size
  __shared__ unsigned int s_rank[1024]; // tile size
  const int tid = threadIdx.x;
  const int start = blockIdx.x * tile;
  const unsigned int mask = (1 << radix_width) - 1; // 1111 (15)

  // coalesced loads into digit
  for( int i = tid; i < tile; i += blockDim.x ) {
    size_t idx = start + i;
    s_digit[i] = ( idx < numElems ) ? (inVals[idx] >> base_bit) & mask : 0xffffffff;
    // 0,1,2,3,0,1,2,3,0,...
  }
  __syncthreads();

  // one thread per digit walks the tile in order -> stable local rank
  if (tid < (1 << radix_width)) {
      unsigned int cnt = 0;
      for (int i = 0; i < tile; ++i) 
          if (s_digit[i] == tid) s_rank[i] = cnt++; // how many times has this block seen this digit?
  }
  __syncthreads();

  for (size_t i = tid; i < tile; i += blockDim.x) {
      size_t idx = start + i;
      if (idx < numElems) {
          unsigned int d   = s_digit[i];
          unsigned int dst = scan[d * gridDim.x + blockIdx.x] + s_rank[i];
          // block's starting offset plus rank seen so far
          outVals[dst] = inVals[idx];
          outPos[dst]  = inPos[idx];
      }
  }
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    size_t tpb = 256; // threads per block
    size_t coarsening_factor = 4; // values histogrammed per thread
    size_t radix_width = 4; // bits histogrammed per step
    size_t hist_blocks = (numElems / coarsening_factor + tpb - 1) / tpb; // 98
    size_t bins_per_block = std::pow(2, radix_width); // 16
    size_t total_bins = hist_blocks * bins_per_block; // 1568

    unsigned int* d_histograms = nullptr;
    cudaMalloc(&d_histograms, sizeof(unsigned int) * total_bins);
    unsigned int* d_scan_output = nullptr;
    cudaMalloc(&d_scan_output, sizeof(unsigned int) * total_bins);

    unsigned int* current_src_vals = d_inputVals;
    unsigned int* current_src_pos = d_inputPos;
    unsigned int* current_dst_vals = d_outputVals;
    unsigned int* current_dst_pos = d_outputPos;

    for( size_t base_bit = 0; base_bit < 32; base_bit += radix_width ) {
      // Each pass should execute:
      // 1) Histogramting of the target bit
      cudaMemset(d_histograms, 0, sizeof(unsigned int) * total_bins);
      histogram_kernel<<<hist_blocks, tpb>>>(
        current_src_vals, numElems, base_bit, radix_width, coarsening_factor, tpb, bins_per_block, d_histograms);
      // 2) single-stage exclusive Scan (Prefix sum) over the histogram
      ex_scan_kernel<<<1, 512>>>(
        d_histograms, total_bins, coarsening_factor, d_scan_output);
      // 3) Predicate mapping & indexing
      // 4) Scattering elements to output buffers

      scatter_kernel<<<hist_blocks, tpb>>>(current_src_vals,
                               current_src_pos,
                               current_dst_vals,
                               current_dst_pos,
                               d_scan_output,
                               numElems,
                               base_bit,
                               radix_width,
                               coarsening_factor * tpb);

      // swap inputs and outputs for next Run
      std::swap( current_src_vals, current_dst_vals );
      std::swap( current_src_pos, current_dst_pos );
    }

    cudaMemcpy(d_outputVals, current_src_vals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_outputPos,  current_src_pos,  numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

    cudaFree(d_histograms);
    cudaFree(d_scan_output);
}

// ============================================================================
// MAIN DRIVER & TESTING UNIT
// ============================================================================
int main() {
    const size_t numElems = 100'000; // Adjust array size to test performance scale
    std::cout << "Allocating and generating test data for " << numElems << " elements...\n";

    // Allocate Host buffers
    std::vector<unsigned int> h_inputVals(numElems);
    std::vector<unsigned int> h_inputPos(numElems);
    std::vector<unsigned int> h_outputVals(numElems);
    std::vector<unsigned int> h_outputPos(numElems);
    std::vector<unsigned int> expected_vals(numElems);
    std::vector<unsigned int> expected_pos(numElems);

    // Seed and generate pseudo-random mock keys/coordinates
    std::mt19937 rng(42);
    std::uniform_int_distribution<unsigned int> dist(0, 1000000);

    for (size_t i = 0; i < numElems; ++i) {
        h_inputVals[i] = dist(rng);
        h_inputPos[i]  = i; // Original positions
    }

    // Run the CPU verification standard
    reference_calculation(h_inputVals.data(), h_inputPos.data(), expected_vals.data(), expected_pos.data(), numElems);

    // Allocate Device memory buffers
    unsigned int *d_inputVals, *d_inputPos, *d_outputVals, *d_outputPos;
    checkCudaErrors(cudaMalloc(&d_inputVals,  numElems * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc(&d_inputPos,   numElems * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc(&d_outputVals, numElems * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc(&d_outputPos,  numElems * sizeof(unsigned int)));

    // Copy initial data to GPU device
    checkCudaErrors(cudaMemcpy(d_inputVals, h_inputVals.data(), numElems * sizeof(unsigned int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_inputPos,  h_inputPos.data(),  numElems * sizeof(unsigned int), cudaMemcpyHostToDevice));

    std::cout << "Launching your_sort implementation...\n";

    // Synchronize and trace computation execution time
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start));
    your_sort(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);
    checkCudaErrors(cudaEventRecord(stop));

    checkCudaErrors(cudaDeviceSynchronize());

    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    // Retrieve sorted GPU output back to Host memory
    checkCudaErrors(cudaMemcpy(h_outputVals.data(), d_outputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_outputPos.data(),  d_outputPos,  numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Check correctness
    bool passed = true;
    for (size_t i = 0; i < numElems; ++i) {
        if (h_outputVals[i] != expected_vals[i] || h_outputPos[i] != expected_pos[i]) {
            std::cerr << "\n❌ MISMATCH DETECTED AT INDEX " << i << "\n";
            std::cerr << "Expected pair: (" << expected_vals[i] << ", " << expected_pos[i] << ")\n";
            std::cerr << "Your GPU pair: (" << h_outputVals[i] << ", " << h_outputPos[i] << ")\n";
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "\n🎉 SUCCESS! All elements correctly sorted parallelly.\n";
        std::cout << "Execution time: " << milliseconds << " ms\n";
    }

    // Cleanup resources
    checkCudaErrors(cudaFree(d_inputVals));
    checkCudaErrors(cudaFree(d_inputPos));
    checkCudaErrors(cudaFree(d_outputVals));
    checkCudaErrors(cudaFree(d_outputPos));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    return passed ? 0 : 1;
}
