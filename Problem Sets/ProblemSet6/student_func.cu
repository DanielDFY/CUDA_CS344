//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */

#include "utils.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>

/* helper device functions */
inline __device__ bool isWithinImg(const int row, const int col, const size_t numRows, const size_t numCols) {
	return ((row >= 0) && (row < numRows) && ((col >= 0) && col < numCols));
}

inline __device__ bool isMasked(const uchar4 val) {
    return (val.x != 255 || val.y != 255 || val.z != 255);
}

inline __device__ int getIdx(const int row, const int col, const size_t numColsSource) {
    return (row * numColsSource + col);
}

/* kernel functions */

// mask generation, as well as the interior and border regions of it
__global__ void generateMaskRegionsKernel(bool* const g_borderRegionOut,
										  bool* const g_interiorRegionOut,
										  const uchar4* const g_imgIn,
										  const size_t numRows,
										  const size_t numCols)
{
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (!isWithinImg(row, col, numRows, numCols))
        return;

    const int idx = getIdx(row, col, numCols);

	if (isMasked(g_imgIn[idx])) {
		// row and col of each neighbor pixel
        int neighborRows[4];
        int neighborCols[4];
        neighborRows[0] = row;
        neighborCols[0] = col - 1;
        neighborRows[1] = row;
        neighborCols[1] = col + 1;
        neighborRows[2] = row - 1;
        neighborCols[2] = col;
        neighborRows[3] = row + 1;
        neighborCols[3] = col;

		// check neighbor pixels
		for (size_t i = 0; i < 4; ++i) {
            const int curRow = neighborRows[i], curCol = neighborCols[i];
			if (isWithinImg(curRow, curCol, numRows, numCols)) {
				if (!isMasked(g_imgIn[getIdx(curRow, curCol, numCols)])) {
                    // one of the neighbors is not in mask region, then current pixel is on border
                    g_borderRegionOut[idx] = true;
                    return;
				}
			}
		}

        // all of the neighbors are in mask region, then current pixel is interior
        g_interiorRegionOut[idx] = true;
	}
}

__global__ void separateChannelsKernel(float* const g_channelROut,
									   float* const g_channelGOut,
									   float* const g_channelBOut,
									   const uchar4* const g_imgIn,
									   const size_t numRows,
									   const size_t numCols)
{
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (!isWithinImg(row, col, numRows, numCols))
        return;

    const int idx = getIdx(row, col, numCols);
	
    const uchar4 value = g_imgIn[idx];

    g_channelROut[idx] = static_cast<float>(value.x);
    g_channelGOut[idx] = static_cast<float>(value.y);
    g_channelBOut[idx] = static_cast<float>(value.z);
}

__global__ void jacobiIterateKernel(float* const g_channelNextOut,
									const float* const g_channelPrevIn,
									const bool* const g_borderRegionIn,
									const bool* const g_interiorRegionIn,
									const float* const g_sourceChannelIn,
									const float* const g_destChannelIn,
									const size_t numRows,
									const size_t numCols)
{
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (!isWithinImg(row, col, numRows, numCols))
        return;

    const int idx = getIdx(row, col, numCols);

	if (g_interiorRegionIn[idx]) {
        float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f;
        const float value = g_sourceChannelIn[idx];

        // row and col of each neighbor pixel
        int neighborRows[4];
        int neighborCols[4];
        neighborRows[0] = row;
        neighborCols[0] = col - 1;
        neighborRows[1] = row;
        neighborCols[1] = col + 1;
        neighborRows[2] = row - 1;
        neighborCols[2] = col;
        neighborRows[3] = row + 1;
        neighborCols[3] = col;

		for (size_t i = 0; i < 4; ++i) {
            const int curRow = neighborRows[i], curCol = neighborCols[i];
            if (isWithinImg(curRow, curCol, numRows, numCols)) {
                ++d;
                const int curIdx = getIdx(curRow, curCol, numCols);
                if (g_interiorRegionIn[curIdx]) {
                    a += g_channelPrevIn[curIdx];
                } else if (g_borderRegionIn[curIdx]) {
                    b += g_destChannelIn[curIdx];
                }
                c += (value - g_sourceChannelIn[curIdx]);
            }
		}

        g_channelNextOut[idx] = min(255.0f, max(0.0, (a + b + c) / d));
	} else {
		// keep original pixel
        g_channelNextOut[idx] = g_destChannelIn[idx];
	}
}

__global__ void recombineChannelsKernel(uchar4* const g_imgOut,
										const float* const g_channelRIn,
										const float* const g_channelGIn,
										const float* const g_channelBIn,
										const size_t numRows,
										const size_t numCols)
{
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (!isWithinImg(row, col, numRows, numCols))
        return;

    const int idx = getIdx(row, col, numCols);

    uchar4 value;
    value.x = g_channelRIn[idx];
    value.y = g_channelGIn[idx];
    value.z = g_channelBIn[idx];

    g_imgOut[idx] = value;
}

/* functions of every step of blending */

void generateMaskRegions(const uchar4* const d_imgIn,
						 const size_t numRows,
						 const size_t numCols,
						 bool* const d_borderRegionOut,
						 bool* const d_interiorRegionOut)
{
    // kernel config
    const dim3 blockSize(32, 32);
    const dim3 gridSize(1 + (numCols - 1)/ blockSize.x, 1 + (numRows - 1) / blockSize.y);

    generateMaskRegionsKernel<<<gridSize, blockSize>>>(d_borderRegionOut, d_interiorRegionOut, d_imgIn, numRows ,numCols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void separateChannels(const uchar4* const d_imgIn,
					  const size_t numRows,
					  const size_t numCols,
					  float* const d_channelROut,
					  float* const d_channelGOut,
					  float* const d_channelBOut)
{
    // kernel config
    const dim3 blockSize(32, 32);
    const dim3 gridSize(1 + (numCols - 1) / blockSize.x, 1 + (numRows - 1) / blockSize.y);

    separateChannelsKernel<<<gridSize, blockSize>>>(d_channelROut, d_channelGOut, d_channelBOut, d_imgIn, numRows, numCols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void jacobiIterate(const float* const d_sourceChannelIn,
				   const float* const d_destChannelIn,
				   const size_t numRows,
				   const size_t numCols,
				   const bool* const d_borderRegionIn,
				   const bool* const d_interiorRegionIn,
				   float* const d_finalChannelOut,
				   const cudaStream_t& stream)
{
    // kernel config
    constexpr size_t ITER_NUM = 800;
	
    const dim3 blockSize(8, 8);
    const dim3 gridSize(1 + (numCols - 1) / blockSize.x, 1 + (numRows - 1) / blockSize.y);

    const size_t channelMemSize = numRows * numCols * sizeof(float);
    float* d_channelPrev, * d_channelNext;

    checkCudaErrors(cudaMalloc(&d_channelPrev, channelMemSize));
    checkCudaErrors(cudaMalloc(&d_channelNext, channelMemSize));

    checkCudaErrors(cudaMemcpyAsync(d_channelPrev, d_sourceChannelIn, channelMemSize, cudaMemcpyDeviceToDevice, stream));

    for (size_t i = 0; i < ITER_NUM; ++i) {
        jacobiIterateKernel<<<gridSize, blockSize, 0, stream>>>(d_channelNext, d_channelPrev, d_borderRegionIn, d_interiorRegionIn, d_sourceChannelIn, d_destChannelIn, numRows, numCols);
        cudaStreamSynchronize(stream); checkCudaErrors(cudaGetLastError());
        std::swap(d_channelPrev, d_channelNext);
	}

    checkCudaErrors(cudaMemcpyAsync(d_finalChannelOut, d_channelPrev, channelMemSize, cudaMemcpyDeviceToDevice, stream));
    cudaStreamSynchronize(stream);

    checkCudaErrors(cudaFree(d_channelPrev));
    checkCudaErrors(cudaFree(d_channelNext));
}

void recombineChannels(const float* const d_finalChannelRIn,
					   const float* const d_finalChannelGIn,
					   const float* const d_finalChannelBIn,
					   const size_t numRows,
					   const size_t numCols,
					   uchar4* const d_finalImgOut)
{
    // kernel config
    const dim3 blockSize(32, 32);
    const dim3 gridSize(1 + (numCols - 1) / blockSize.x, 1 + (numRows - 1) / blockSize.y);

    recombineChannelsKernel<<<gridSize, blockSize>>>(d_finalImgOut, d_finalChannelRIn, d_finalChannelGIn, d_finalChannelBIn, numRows, numCols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{
    /*
      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
	*/
	
    const size_t imgSize = numRowsSource * numColsSource;
    const size_t imgMemSize = imgSize * sizeof(uchar4);

    uchar4* d_sourceImg, * d_destImg;

    checkCudaErrors(cudaMalloc(&d_sourceImg, imgMemSize));
    checkCudaErrors(cudaMalloc(&d_destImg, imgMemSize));

    checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, imgMemSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, imgMemSize, cudaMemcpyHostToDevice));

    /*
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.
    */

    const size_t regionMemSize = imgSize * sizeof(bool);
    bool* d_borderRegion,* d_interiorRegion;

    checkCudaErrors(cudaMalloc(&d_borderRegion, regionMemSize));
    checkCudaErrors(cudaMalloc(&d_interiorRegion, regionMemSize));

    checkCudaErrors(cudaMemset(d_borderRegion, false, regionMemSize));
    checkCudaErrors(cudaMemset(d_interiorRegion, false, regionMemSize));

    generateMaskRegions(d_sourceImg, numRowsSource, numColsSource, d_borderRegion, d_interiorRegion);

    /*
     3) Separate out the incoming image into three separate channels
    */

    const size_t channelMemSize = imgSize * sizeof(float);
    float* d_sourceChannelR, * d_sourceChannelG, * d_sourceChannelB;
    float* d_destChannelR, * d_destChannelG, * d_destChannelB;

    checkCudaErrors(cudaMalloc(&d_sourceChannelR, channelMemSize));
    checkCudaErrors(cudaMalloc(&d_sourceChannelG, channelMemSize));
    checkCudaErrors(cudaMalloc(&d_sourceChannelB, channelMemSize));

    checkCudaErrors(cudaMalloc(&d_destChannelR, channelMemSize));
    checkCudaErrors(cudaMalloc(&d_destChannelG, channelMemSize));
    checkCudaErrors(cudaMalloc(&d_destChannelB, channelMemSize));

    separateChannels(d_sourceImg, numRowsSource, numColsSource, d_sourceChannelR, d_sourceChannelG, d_sourceChannelB);
    separateChannels(d_destImg, numRowsSource, numColsSource, d_destChannelR, d_destChannelG, d_destChannelB);

    /*
     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our initial guess.
        
     5) For each color channel perform the Jacobi iteration described 
        above 800 times.
    */
	
    float* d_finalChannelR, * d_finalChannelG, * d_finalChannelB;
    checkCudaErrors(cudaMalloc(&d_finalChannelR, channelMemSize));
    checkCudaErrors(cudaMalloc(&d_finalChannelG, channelMemSize));
    checkCudaErrors(cudaMalloc(&d_finalChannelB, channelMemSize));

    cudaStream_t streamChannelR, streamChannelG, streamChannelB;
    checkCudaErrors(cudaStreamCreateWithFlags(&streamChannelR, cudaStreamNonBlocking));
    checkCudaErrors(cudaStreamCreateWithFlags(&streamChannelG, cudaStreamNonBlocking));
    checkCudaErrors(cudaStreamCreateWithFlags(&streamChannelB, cudaStreamNonBlocking));
	
    jacobiIterate(d_sourceChannelR, d_destChannelR, numRowsSource, numColsSource, d_borderRegion, d_interiorRegion, d_finalChannelR, streamChannelR);
    jacobiIterate(d_sourceChannelG, d_destChannelG, numRowsSource, numColsSource, d_borderRegion, d_interiorRegion, d_finalChannelG, streamChannelG);
    jacobiIterate(d_sourceChannelB, d_destChannelB, numRowsSource, numColsSource, d_borderRegion, d_interiorRegion, d_finalChannelB, streamChannelB);

    cudaDeviceSynchronize();

    checkCudaErrors(cudaStreamDestroy(streamChannelR));
    checkCudaErrors(cudaStreamDestroy(streamChannelG));
    checkCudaErrors(cudaStreamDestroy(streamChannelB));

	/*
	 6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.
	 */

    uchar4* d_blendedImg;
    checkCudaErrors(cudaMalloc(&d_blendedImg, imgMemSize));

    recombineChannels(d_finalChannelR, d_finalChannelG, d_finalChannelB, numRowsSource, numColsSource, d_blendedImg);

    checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, imgMemSize, cudaMemcpyDeviceToHost));

	/*
	 * clean up
	 */
    checkCudaErrors(cudaFree(d_sourceImg));
    checkCudaErrors(cudaFree(d_destImg));
	
    checkCudaErrors(cudaFree(d_borderRegion));
    checkCudaErrors(cudaFree(d_interiorRegion));
	
    checkCudaErrors(cudaFree(d_sourceChannelR));
    checkCudaErrors(cudaFree(d_sourceChannelG));
    checkCudaErrors(cudaFree(d_sourceChannelB));
	
    checkCudaErrors(cudaFree(d_destChannelR));
    checkCudaErrors(cudaFree(d_destChannelG));
    checkCudaErrors(cudaFree(d_destChannelB));
	
    checkCudaErrors(cudaFree(d_finalChannelR));
    checkCudaErrors(cudaFree(d_finalChannelG));
    checkCudaErrors(cudaFree(d_finalChannelB));
	
    checkCudaErrors(cudaFree(d_blendedImg));
}
