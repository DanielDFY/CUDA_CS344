/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"
#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))

#define WARP_SIZE 32

// global variable: 0 unlocked, 1 locked
__device__ int d_min_max_lock = 0;

__constant__ size_t d_warp_per_block;

__global__ void find_min_max_kernel(float2* g_min_max_out, const float* g_logLuminance_in, const size_t data_size) {
    extern __shared__ float2 min_max_shared[];

    const int tid = threadIdx.x;
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int lid = tid % WARP_SIZE;
    const int wid = tid / WARP_SIZE;

    float minValue, maxValue;
	if (idx < data_size) {
        minValue = g_logLuminance_in[idx];
        maxValue = g_logLuminance_in[idx];
	} else {
        minValue = FLT_MAX;
        maxValue = FLT_MIN;
        return;
	}

	// find the min and max value in each warp
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        minValue = min(minValue, __shfl_down_sync(-1, minValue, offset));
        maxValue = max(maxValue, __shfl_down_sync(-1, maxValue, offset));
    }

    if (lid == 0) {
        min_max_shared[wid] = {minValue, maxValue};
    }

    __syncthreads();

    // find the min and max value in each block
    if (tid < WARP_SIZE) {
		if (tid < d_warp_per_block) {
            minValue = min_max_shared[tid].x;
            maxValue = min_max_shared[tid].y;
		} else {
            minValue = FLT_MAX;
            maxValue = FLT_MIN;
		}
    }

    if (wid == 0) {
        for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            minValue = min(minValue, __shfl_down_sync(-1, minValue, offset));
            maxValue = max(maxValue, __shfl_down_sync(-1, maxValue, offset));
        }
    }

    // find the min and max value among all blocks
    if (tid == 0) {
        do {} while (atomicCAS(&d_min_max_lock, 0, 1));
        (*g_min_max_out).x = min((*g_min_max_out).x, minValue);
        (*g_min_max_out).y = max((*g_min_max_out).y, maxValue);
        d_min_max_lock = 0;
    }
}

__global__ void histogram_kernel(unsigned int* const g_bins_out, const float* const g_logLuminance_in, const size_t data_size,
    const float min_logLum, const float range_logLum, const size_t numBins) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= data_size)
        return;

	// formula: bin = (lum[i] - lumMin) / lumRange * numBins
    int bin = ((g_logLuminance_in[idx] - min_logLum) / range_logLum) * numBins;

    atomicAdd(&g_bins_out[bin], 1);
}

__global__ void exclusive_scan_kernel(unsigned int* const g_cdf, unsigned int* g_bins, int size) {
    extern __shared__  unsigned int scan_shared[];

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;
	
    int tid = threadIdx.x;
    unsigned int temp = g_bins[idx];
    scan_shared[tid] = temp;

	// exclusive scan of each block
    for (int d = 1; d < blockDim.x; d = 2 * d) {
        __syncthreads();

        if (tid - d >= 0) temp = temp + scan_shared[tid - d];

        __syncthreads();

        scan_shared[tid] = temp;
    }

    __syncthreads();

    temp = scan_shared[tid];

    if (idx < size - 1)
		g_cdf[idx + 1] = temp;
}

__global__ void add_pre_sum_kernel(unsigned int* g_cdf, unsigned int* g_pre_sum, const size_t size) {
    // add sum of precious blocks to each element in current block to get final exclusive scan
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= size)
        return;
	
    if (tid != 0)
        g_cdf[tid] += g_pre_sum[(tid - 1) / blockDim.x];
}

float2 find_min_max(const float* const d_logLuminance, const size_t data_size) {
	// kernel config
    const size_t thread_per_block = 1024;
    const size_t num_block = 1 + (data_size - 1) / thread_per_block;

	// set constant variable and shared memory size
    const size_t h_warp_per_block = 1 + (thread_per_block - 1) / WARP_SIZE;
    checkCudaErrors(cudaMemcpyToSymbol(d_warp_per_block, &h_warp_per_block, sizeof(h_warp_per_block)));
    const size_t shared_mem_size = sizeof(float2) * h_warp_per_block;

	// get min and max of luminance data
    float2* h_min_max_out;
    h_min_max_out = (float2*)malloc(sizeof(float2));
    *h_min_max_out = { FLT_MAX, FLT_MIN };
    float2* d_min_max_out;
    checkCudaErrors(cudaMalloc(&d_min_max_out, sizeof(float2)));
    checkCudaErrors(cudaMemcpy(d_min_max_out, h_min_max_out, sizeof(float2), cudaMemcpyHostToDevice));

    find_min_max_kernel<<<num_block, thread_per_block, shared_mem_size>>>(d_min_max_out, d_logLuminance, data_size);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(h_min_max_out, d_min_max_out, sizeof(float2), cudaMemcpyDeviceToHost));

    float min = (*h_min_max_out).x;
    float max = (*h_min_max_out).y;

    free(h_min_max_out);
    checkCudaErrors(cudaFree(d_min_max_out));

	if (min > max) {
        throw std::runtime_error("invalid log luminance input");
	} else {
        return { min, max };
	}
}

void generate_histogram(const float* const d_logLuminance, unsigned int* const d_bins, const size_t data_size,
    const float min_logLum, const float range_logLum, const size_t numBins) {
    // kernel config
    const size_t thread_per_block = 1024;
    const size_t num_block = 1 + (data_size - 1) / thread_per_block;

    histogram_kernel<<<num_block, thread_per_block>>>(d_bins, d_logLuminance, data_size, min_logLum, range_logLum, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void exclusive_scan(unsigned int* const d_cdf, unsigned int* const d_bins, const size_t numBins) {
    // Hillis Steele Scan supporting multi-blocks
	
    // kernel config
    const size_t thread_per_block = 1024;
    const size_t num_block = 1 + (numBins - 1) / thread_per_block;

	// do local scan for each block
    const size_t shared_mem_size = thread_per_block * sizeof(unsigned int);
    exclusive_scan_kernel<<<num_block, thread_per_block, shared_mem_size>>>(d_cdf, d_bins, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    const size_t histo_mem_size = numBins * sizeof(unsigned int);
    unsigned int* h_cdf;
	h_cdf = (unsigned int*)malloc(histo_mem_size);
    checkCudaErrors(cudaMemcpy(h_cdf, d_cdf, histo_mem_size, cudaMemcpyDeviceToHost));

	// sum local scan of previous blocks for each block 
    const size_t pre_sum_mem_size = num_block * sizeof(unsigned int);
    unsigned int* h_pre_sum;
	h_pre_sum = (unsigned int*)malloc(pre_sum_mem_size);
    unsigned int pre_sum = 0;
    for (int i = 0; i < num_block - 1; ++i) {
        h_pre_sum[i] = pre_sum;
        pre_sum += h_cdf[(i + 1) * thread_per_block];
    }
    h_pre_sum[num_block - 1] = pre_sum;

    unsigned int* d_pre_sum;
    checkCudaErrors(cudaMalloc(&d_pre_sum, pre_sum_mem_size));
    checkCudaErrors(cudaMemcpy(d_pre_sum, h_pre_sum, pre_sum_mem_size, cudaMemcpyHostToDevice));

	// add local scan of previous blocks to get global scan
    add_pre_sum_kernel<<<num_block, thread_per_block>>>(d_cdf, d_pre_sum, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    free(h_cdf);
    free(h_pre_sum);
    checkCudaErrors(cudaFree(d_pre_sum));
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
	unsigned int* const d_cdf, float& min_logLum, float& max_logLum,
	const size_t numRows, const size_t numCols, const size_t numBins) {
	
    //TODO
	/* Here are the steps you need to implement */
    
    const size_t img_data_size = numCols * numRows;

	/* 1) find the minimum and maximum value in the input logLuminance channel
          store in min_logLum and max_logLum */
	
    float2 min_max = find_min_max(d_logLuminance, img_data_size);
    min_logLum = min_max.x;
    max_logLum = min_max.y;

    /* 2) subtract them to find the range */
	
    float range_logLum = max_logLum - min_logLum;

    /* 3) generate a histogram of all the values in the logLuminance channel using
          the formula: bin = (lum[i] - lumMin) / lumRange * numBins */

	// initialize histogram bins
    unsigned int* d_bins;
    const size_t histo_mem_size = numBins * sizeof(unsigned int);
    checkCudaErrors(cudaMalloc(&d_bins, histo_mem_size));
    checkCudaErrors(cudaMemset(d_bins, 0, histo_mem_size));
	
    generate_histogram(d_logLuminance, d_bins, img_data_size, min_logLum, range_logLum, numBins);

    /* 4) Perform an exclusive scan (prefix sum) on the histogram to get
          the cumulative distribution of luminance values (this should go in the
          incoming d_cdf pointer which already has been allocated for you) */

    exclusive_scan(d_cdf, d_bins, numBins);

    checkCudaErrors(cudaFree(d_bins));
}
