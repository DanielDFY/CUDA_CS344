/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#include "device_launch_parameters.h"

#define THREADS_PER_BLOCK 1024
#define VALS_PER_THREAD 256

__global__
void baseline(const unsigned int* const g_vals, //INPUT
			  unsigned int* const g_histo,      //OUPUT
			  const unsigned int numElems) {
	
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= numElems)
        return;

    atomicAdd((g_histo + g_vals[id]), 1);
}

__global__ void yourHisto(const unsigned int* const g_vals, //INPUT
						  unsigned int* const g_histo,      //OUPUT
						  const unsigned int numElems) {

    extern __shared__ unsigned int histo_shared[];

    int tid = threadIdx.x;
	
    histo_shared[tid] = 0;
    __syncthreads();

    int idx = blockDim.x * (VALS_PER_THREAD * blockIdx.x) + threadIdx.x;
	
    for (int i = 0; i < VALS_PER_THREAD; ++i, idx += blockDim.x) {
        if (idx < numElems) {
            atomicAdd(&histo_shared[g_vals[idx]], 1);
        }
    }

    __syncthreads();

    atomicAdd(&g_histo[threadIdx.x], histo_shared[threadIdx.x]);
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
					  unsigned int* const d_histo,      //OUTPUT
					  const unsigned int numBins,
					  const unsigned int numElems) {
	
    // const int thread_per_block_base = THREADS_PER_BLOCK;
    // const int num_block_base = 1 + (numElems - 1) / thread_per_block_base;

    const int thread_per_block_yourHisto = THREADS_PER_BLOCK;
    const int num_block_yourHisto = 1 + (numElems - 1) / (VALS_PER_THREAD * thread_per_block_yourHisto);

    // baseline<<<num_block_base, thread_per_block_base>>>(d_vals, d_histo, numElems);
    yourHisto<<<num_block_yourHisto, thread_per_block_yourHisto, sizeof(unsigned int)* thread_per_block_yourHisto>>> (d_vals, d_histo, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}