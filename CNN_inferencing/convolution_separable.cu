#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////

__constant__ float d_Kernel[KERNEL_LENGTH];

__global__ void convolutionRowGPU(
                  float *d_Result,
                  float *d_Data,
                  int dataW,
                  int dataH
                 )
{
  // Data cache: threadIdx.x , threadIdx.y
  __shared__ float data[TILE_W + KERNEL_RADIUS * 2][TILE_H];

  // global mem address of this thread
  const int gLoc = threadIdx.x +
            IMUL(blockIdx.x, blockDim.x) +
            IMUL(threadIdx.y, dataW) +
            IMUL(blockIdx.y, blockDim.y) * dataW;

  int x; // image based coordinate

  // original image based coordinate
  const int x0 = threadIdx.x + IMUL(blockIdx.x, blockDim.x);

   // case1: left
  x = x0 - KERNEL_RADIUS;
  if ( x < 0 )
    data[threadIdx.x][threadIdx.y] = 0;
  else
    data[threadIdx.x][threadIdx.y] = d_Data[ gLoc - KERNEL_RADIUS];

   // case2: right
  x = x0 + KERNEL_RADIUS;
  if ( x > dataW-1 )
    data[threadIdx.x + blockDim.x][threadIdx.y] = 0;
  else
    data[threadIdx.x + blockDim.x][threadIdx.y] = d_Data[gLoc + KERNEL_RADIUS];

   __syncthreads();

  // convolution
  float sum = 0;
  x = KERNEL_RADIUS + threadIdx.x;
  
  for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)
    sum += data[x + i][threadIdx.y] * d_Kernel[KERNEL_RADIUS + i];

   d_Result[gLoc] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColGPU(
                  float *d_Result,
                  float *d_Data,
                  int dataW,
                  int dataH
                .)
{
  // Data cache: threadIdx.x , threadIdx.y
  __shared__ float data[TILE_W][TILE_H + KERNEL_RADIUS * 2];

   // global mem address of this thread
  const int gLoc = threadIdx.x +
            IMUL(blockIdx.x, blockDim.x) +
            IMUL(threadIdx.y, dataW) +
            IMUL(blockIdx.y, blockDim.y) * dataW;

  int y; // image based coordinate

  // original image based coordinate
  const int y0 = threadIdx.y + IMUL(blockIdx.y, blockDim.y);

  // case1: upper
  y = y0 - KERNEL_RADIUS;
  if ( y < 0 )
    data[threadIdx.x][threadIdx.y] = 0;
  else
    data[threadIdx.x][threadIdx.y] = d_Data[ gLoc - IMUL(dataW, KERNEL_RADIUS)];

  // case2: lower
  y = y0 + KERNEL_RADIUS;
  if ( y > dataH-1 )
    data[threadIdx.x][threadIdx.y + blockDim.y] = 0;
  else
    data[threadIdx.x][threadIdx.y + blockDim.y] = d_Data[gLoc + IMUL(dataW, KERNEL_RADIUS)];

  __syncthreads();

  // convolution
  float sum = 0;
  y = KERNEL_RADIUS + threadIdx.y;
  for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)
    sum += data[threadIdx.x][y + i] * d_Kernel[KERNEL_RADIUS + i];

  d_Result[gLoc] = sum;
}