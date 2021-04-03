#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <time.h>
#include <math.h>
#include <algorithm>


//For the given architecture, we do not require strides in avePooling layer. each N*M matrix is converted into one single value and stored into a memory location
//It is a simple reduction kernel applied to smaller segments of an array < 32 generally. Even if the layer is large, say 16*16, we can launch that many threads per block
/*_device__ void warpReduce ( int * sdata , int tid ) 
{
    sdata [tid] += sdata [tid + 32];
    sdata [tid] += sdata [tid + 16];
    sdata [tid] += sdata [tid + 8];
    sdata [tid] += sdata [tid + 4];
    sdata [tid] += sdata [tid + 2];
    sdata [tid] += sdata [tid + 1];
}*/
//Misaligned data acceses are not as problematic in modern GPUs due to larger L1 Cache width
//https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/#:~:text=Misaligned%20Data%20Accesses&text=Arrays%20allocated%20in%20device%20memory,are%20aligned%20to%20their%20size.

__global__ void avgPoolKernel(float*A, float*B, int threadsPerMat, int nearest_2)
{   
    //threadsPerMat is equal to size of each matrix
    //__shared__ tile[size];  //Make a very very large one to be true, large enough to load all elements per block
    extern __shared__ float tile[]; //For now, one block computes one matrix only
    unsigned int tid,local_tid,access_id;
    //local_tid = threadIdx.y*threadDim.x + threadIdx.x;
    local_tid = threadIdx.x;
    //tid = blockIdx.x*blockDim.x*blockDim.y + local_tid;
    tid = blockIdx.x*blockDim.x + local_tid;
    access_id = local_tid + blockIdx.x*threadsPerMat;
    //printf("local_tid: %d, tid: %d\n",local_tid,tid);
    //printf("local_tid: %d, tid: %d, tile[local_tid]: %f\n",local_tid,tid,tile[local_tid]);
    
    //Note that shared memory in CUDA is not initialized!!!

    if(local_tid < threadsPerMat - nearest_2)
    {
        tile[local_tid] = A[access_id + nearest_2];
    }
    else
    {
        tile[local_tid] = 0;
    }
    //printf("local_tid: %d, tid: %d, tile[local_tid]: %f\n",local_tid,tid,tile[local_tid]);
    
    tile[local_tid] += A[access_id];
    __syncthreads();
    //printf("local_tid: %d, tid: %d, tile[local_tid]: %f\n",local_tid,tid,tile[local_tid]);
    //All the elements are loaded and are now done to closest power of 2

    for (unsigned int i = nearest_2/2; i >0 ; i>>=1)
    {
        if(local_tid<i)
        {
            tile[local_tid] += tile[local_tid + i]; 
        }
        __syncthreads();
    }
    // if (local_tid < 32)
    // {
    //     warpReduce(tile,local_tid);
    // }

    //As of now, one block calculates one full tile
    if(local_tid==0)
    {
        B[blockIdx.x] = tile[0]/threadsPerMat;
        // printf("Tile: %f, Block: %d\n",tile[0],blockIdx.x);
    }


    // tile[block_tid] = A[tid];   //block_tid is id of the matrix in the block conatining channels number of channels
    // int mat_tid = tid%threadsPerMat;
    // unsigned int i = threadsPerMat;
    // for (i = threadsPerMat; i>0; )
    // {
    //     if((i & 0x01) == 1) //If i is odd, we need to compensate for that loner at the end
    //     {
    //         if(tid == 0)
    //         {
    //            tile[tid] += tile[tid-1 + threadsPerMat];
    //         }
    //     }
    //     i>>=1;  //Divide i by 2
    //     if(mat_tid<i)
    //     {
    //         tile[tid] += tile[tid+i];
    //     }
    //     if(mat_tid==0 )
    //     {
    //         tile[tid] +=tile[]
    //     }
    //     __syncthreads();

    // }
    //Access Threads per mat each time and keep on dividing it by two till you reach 1.
    //Divide by ThreadsperMat at the end, and return in the array required.
}


//We try to access memory in strides, but we have to put an if condition to write in proper locations. 
//i.e. we did not have too many if conditions before, but uncoalesced memory accesses. 
//Now we will have too many if else conditions initially.
//We can assume that the inputs are of small size, <12X12 or even 15X15. In this case we will try to club as many blocks as we can such that we get full 32.

// __global__ avgPoolKernelV2(float*A, float*B, int threadsPerMat, int nearest_2)
// {

// }

void avgPool(int width, int height, int channels, float* hA, float* hB, float* dA, float* dB)
{
    //struct cudeDeviceProp devp;
    //cudaGetDeviceProperties(&devp,0);
    //int maxThreadsPerBlock = devp.maxThreadsPerBlock;
    //int maxX = devp.maxThreadsDim[0], maxY = devp.maxThreadsDim[1], maxZ = devp.maxThreadsDim[2];
    //int sharedMemPerBlock = devp.sharedMemPerBlock;
    int sharedMemPerBlock = 49152;
    int maxThreadsPerBlock = 1024;
    int maxX = 1024, maxY = 1024, maxZ = 64;
    
    int threadsPerMat = width*height;
    int nearest_2 = pow(2,floor(log2(threadsPerMat)));
    int size = nearest_2;

    dim3 gridSize(channels,1,1);
    dim3 blockSize(nearest_2,1);

    cudaError_t err = cudaSuccess;
    //If I can not access dA if it exists on the device, we can replace dA with hA and make them mutually exclusive
    //If hA is null, dA must have the data, if hA is not null, dA has to be allocated
    //dA == NULL works even if it is allocated in GPU MEM, it was tested
    if(dA == NULL)
    {
        err = cudaMalloc((void**)&dA,height*width*channels*sizeof(float));
        if(err !=cudaSuccess)
        {
            fprintf(stderr, "avgPool: Failed to allocate device vector A (code:%s)\n",cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        if(hA == NULL)
        {
            fprintf(stderr, "No input A (device nor host) Given to avgPool\n");
            exit(0);
        }
        err = cudaMemcpy(dA, hA, height*width*channels*sizeof(float) , cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

    }

    // if(dB == NULL)
    // {
        err = cudaMalloc((void**)&dB,channels*sizeof(float));
        if(err !=cudaSuccess)
        {
            fprintf(stderr, "avgPool: Failed to allocate device vector B (code:%s)\n",cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }        
    // }





    avgPoolKernel<<<gridSize,blockSize,nearest_2*sizeof(float)>>>(dA,dB,threadsPerMat,nearest_2);

    if(hB!=NULL)
    {
        err = cudaMemcpy(hB, dB, channels*sizeof(float) , cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy device vector B from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }        
    }

    //Note: The driver code has to free both hA, hB, dA, dB if they are not required.

    // int dimx, dimy, dimz, tot_data;
    
    // tot_data = width*height*channels;
    // dimx = fmin(width,maxX);
    // dimy = fmin(height,maxY);
    // dimz = fmin(maxZ,tot_data/dimx/dimy);   //This is generally equal to channels, but if the width is too high, then it becomes smaller than channels
    // dimz = fmin(dimz, sharedMemPerBlock/dimx/dimy/4);  //Gives the maximum width across X such that all of the memory spaces are utilized in each block
    
    // dim3 blockSize(dimx,dimy,dimz);

    //Blocks arranged across Z axis as well only.
    // size_t dimGz = tot_data/dimy/dimx/dimz;

   // dim3 gridSize(1,1,dimGz);


    //printf("dimx: %d, dimy: %d, dimz: %d, blocks: %d ",dimx,dimy,dimz,dimGz);

    //int dimz = min(maxZ,maxThreadsPerBlock/maxX/maxY,channels,);
}

int main(int argc, char const *argv[])
{
    //Driver Code only
    int channels = 1024, height = 7, width = 7;
    float *hA = (float *)malloc(width*height*channels*sizeof(float));
    float *hB = (float *)malloc(channels*sizeof(float));
    float *dA = NULL;
    float *dB = NULL;

    for (int k = 0; k < channels; ++k)
    {
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                hA[k*width*height + i*width + j] = j*(k+1);
                // printf("%d ", j*(k+1));
            }
            // printf("\n");
        }        

        // printf("\nChannel End\n");
    }
 
 //Testing if dA == NULL can be done on host if dA is on Device   
    // cudaError_t err = cudaSuccess;
    // err = cudaMalloc((void**)&dA,height*width*channels*sizeof(float));
    // if(err !=cudaSuccess)
    // {
    //     fprintf(stderr, "avgPool: Failed to allocate device vector A (code:%s)\n",cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // err = cudaMemcpy(dA, hA, height*width*channels*sizeof(float) , cudaMemcpyHostToDevice);
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    avgPool(width,height,channels,hA,hB,dA,dB);

    // for (int k = 0; k < channels; ++k)
    // {
    //     printf("%f\n", *(hB+k));
    // }

    printf("DONE!!!\n");

    return 0;
}