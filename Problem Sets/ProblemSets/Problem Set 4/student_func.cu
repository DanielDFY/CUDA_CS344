//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__ void histogram_kernel(unsigned int* const g_outputBins,
								 const unsigned int* const g_inputVals,
								 const size_t input_size,
								 const int digit_offset) {
	// histogram of the number of occurrences of each digit at position with given offset (start from LSB)
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= input_size)
        return;

	// check if target digit of current data number is 1
    bool isOne = (g_inputVals[idx] & (1 << digit_offset)) != 0;
	
    if (isOne)
        atomicAdd(&g_outputBins[1], 1);     // digit is 1, increase bin[1]
    else
        atomicAdd(&g_outputBins[0], 1);     // digit is 0, increase bin[0]
}

__global__ void exclusive_scan_kernel(unsigned int* const g_outputPrefixSum,
									  const unsigned int* g_inputVals,
									  const size_t input_size,
									  const int digit_offset) {
    // determine relative offset of each digit
	
    extern __shared__  unsigned int scan_shared[];

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= input_size)
        return;

    int tid = threadIdx.x;
    unsigned int val = ((g_inputVals[idx] & (1 << digit_offset)) != 0) ? 1 : 0;
    scan_shared[tid] = val;

    // exclusive scan of each block
    for (int d = 1; d < blockDim.x; d = 2 * d) {
        __syncthreads();

        if (tid - d >= 0) val = val + scan_shared[tid - d];

        __syncthreads();

        scan_shared[tid] = val;
    }

    __syncthreads();

    val = scan_shared[tid];

    if (idx < input_size - 1)
        g_outputPrefixSum[idx + 1] = val;
}

__global__ void add_pre_sum_kernel(unsigned int* g_outputFinalPrefixSum,const unsigned int* g_inputBlockPrefixSum, const size_t input_size) {
    // add sum of precious blocks to each element in current block to get final exclusive scan
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= input_size)
        return;
	
    if (tid != 0)
        g_outputFinalPrefixSum[tid] += g_inputBlockPrefixSum[(tid - 1) / blockDim.x];
}

__global__ void move_kernel(unsigned int* const g_outputVals,
							unsigned int* const g_outputPos,
							const unsigned int* const g_inputVals,
							const unsigned int* const g_inputPos,
							const unsigned int* const g_inputPrefixSum,
							const size_t input_size,
							const unsigned int one_begin_idx,
							const int digit_offset) {
	// determine the final output location for each element and move it there
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= input_size)
        return;

    bool isOne = (g_inputVals[idx] & (1 << digit_offset)) != 0;

	// if digit is 0, final position start from 0
    // if digit is 1, final position start from the end of all the 0s
    unsigned int base = isOne ? one_begin_idx : 0;

	// inputPrefixSum refers to prefix sum of 1s, so prefix sum of 0s should be (idx - inputPrefixSum)
    unsigned int prefixSum = isOne ? g_inputPrefixSum[idx] : (idx - g_inputPrefixSum[idx]);

	// final position should be (base + prefixSum)
    int pos = base + prefixSum;
    g_outputVals[pos] = g_inputVals[idx];
    g_outputPos[pos] = g_inputPos[idx];
}

void generate_histogram(const unsigned int* const d_inputVals,
						unsigned int* const d_outputBins,
						const size_t input_size,
						const int digit_offset) {
    // kernel config
    const size_t thread_per_block = 1024;
    const size_t num_block = 1 + (input_size - 1) / thread_per_block;

	// generate histogram bins
    histogram_kernel<<<num_block, thread_per_block>>>(d_outputBins, d_inputVals, input_size, digit_offset);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void exclusive_scan(const unsigned int* const d_inputVals,
					unsigned int* const d_outputPrefixSum,
					const size_t input_size,
					const int digit_offset) {
    // Hillis Steele Scan supporting multi-blocks

    // kernel config
    const size_t thread_per_block = 1024;
    const size_t num_block = 1 + (input_size - 1) / thread_per_block;

    // do local scan for each block
    const size_t shared_mem_size = thread_per_block * sizeof(unsigned int);
    exclusive_scan_kernel<<<num_block, thread_per_block, shared_mem_size>>>(d_outputPrefixSum, d_inputVals, input_size, digit_offset);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    const size_t histo_mem_size = input_size * sizeof(unsigned int);
    unsigned int* h_outputPrefixSum;
    h_outputPrefixSum = (unsigned int*)malloc(histo_mem_size);
    checkCudaErrors(cudaMemcpy(h_outputPrefixSum, d_outputPrefixSum, histo_mem_size, cudaMemcpyDeviceToHost));

    // sum local scan of previous blocks for each block 
    const size_t pre_sum_mem_size = num_block * sizeof(unsigned int);
    unsigned int* h_pre_sum;
    h_pre_sum = (unsigned int*)malloc(pre_sum_mem_size);
    unsigned int pre_sum = 0;
    for (int i = 0; i < num_block - 1; ++i) {
        h_pre_sum[i] = pre_sum;
        pre_sum += h_outputPrefixSum[(i + 1) * thread_per_block];
    }
    h_pre_sum[num_block - 1] = pre_sum;

    unsigned int* d_pre_sum;
    checkCudaErrors(cudaMalloc(&d_pre_sum, pre_sum_mem_size));
    checkCudaErrors(cudaMemcpy(d_pre_sum, h_pre_sum, pre_sum_mem_size, cudaMemcpyHostToDevice));

    // add local scan of previous blocks to get global scan
    add_pre_sum_kernel<<<num_block, thread_per_block>>>(d_outputPrefixSum, d_pre_sum, input_size);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //checkCudaErrors(cudaMemcpy(h_outputPrefixSum, d_outputPrefixSum, histo_mem_size, cudaMemcpyDeviceToHost));
	//printf("student: %d %d\n", h_outputPrefixSum[0], h_outputPrefixSum[1]);

    free(h_outputPrefixSum);
    free(h_pre_sum);
    checkCudaErrors(cudaFree(d_pre_sum));
}

void move(unsigned int* const d_inputVals,
		  unsigned int* const d_inputPos,
		  unsigned int* const d_outputVals,
		  unsigned int* const d_outputPos,
		  const unsigned int* const d_inputPrefixSum,
		  const size_t input_size,
		  const unsigned int one_begin_idx,
		  const int digit_offset) {
    // kernel config
    const size_t thread_per_block = 1024;
    const size_t num_block = 1 + (input_size - 1) / thread_per_block;

	// move each input value to its final position
    move_kernel<<<num_block, thread_per_block>>>(d_outputVals, d_outputPos, d_inputVals, d_inputPos, d_inputPrefixSum, input_size, one_begin_idx, digit_offset);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems) { 
	//TODO
	//PUT YOUR SORT HERE

	// initialize histogram bins
    const size_t histo_mem_size = 2 * sizeof(unsigned int);

    unsigned int* h_bins;
    h_bins = (unsigned int*)malloc(histo_mem_size);
	
    unsigned int* d_bins;
    checkCudaErrors(cudaMalloc(&d_bins, histo_mem_size));

    const size_t input_mem_size = numElems * sizeof(unsigned int);

    // initialize prefix sum array
    unsigned int* d_prefixSum;
    const size_t prefixSum_mem_size = input_mem_size;
    checkCudaErrors(cudaMalloc(&d_prefixSum, prefixSum_mem_size));

	// do radix sort
    constexpr unsigned int value_digits = 8 * sizeof(unsigned int);
	for (unsigned int digit_offset = 0; digit_offset < value_digits; ++digit_offset) {
		checkCudaErrors(cudaMemset(d_bins, 0, histo_mem_size));
        checkCudaErrors(cudaMemset(d_prefixSum, 0, prefixSum_mem_size));

        generate_histogram(d_inputVals, d_bins, numElems, digit_offset);
        exclusive_scan(d_inputVals, d_prefixSum, numElems, digit_offset);
        checkCudaErrors(cudaMemcpy(h_bins, d_bins, histo_mem_size, cudaMemcpyDeviceToHost));
        move(d_inputVals, d_inputPos, d_outputVals, d_outputPos, d_prefixSum, numElems, h_bins[0], digit_offset);

        checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, input_mem_size, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_inputPos, d_outputPos, input_mem_size, cudaMemcpyDeviceToDevice));
	}

	// copy result to output
    checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, input_mem_size, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, input_mem_size, cudaMemcpyDeviceToDevice));

    free(h_bins);
    checkCudaErrors(cudaFree(d_bins));
    checkCudaErrors(cudaFree(d_prefixSum));
}
