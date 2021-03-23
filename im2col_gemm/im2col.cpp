%%cuda --name im2col.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
     
   

__global__ void Im2Col_optimised(float *A, float *B,int h,int w,int c,int s)
{
    int blockNum = blockIdx.z * (gridDim.y * gridDim.x) + blockIdx.x * gridDim.y +blockIdx.y;
    int threadNum = threadIdx.z * (blockDim.y * blockDim.x) + threadIdx.x * (blockDim.y) + threadIdx.y;
    int globalThreadId = blockNum * (blockDim.y * blockDim.x * blockDim.z) + threadNum;

    int converted_h=h-s+1;
    int converted_w=w-s+1;

    int k=globalThreadId/(h*w);
    int i=(globalThreadId%(h*w))/w;
    int j=(globalThreadId%(h*w))%w;
   
    __shared__  float arr[1024];
    int tid=threadIdx.x;

    arr[tid]=A[globalThreadId];
    __syncthreads();

    //printf("%d %d %d %d\n",globalThreadId,i,j,k);

    blockNum%=h;

    int l=0;
    int r=w-s;

    int rangeup=s*s*k;
    int rangedn=s*s*(k+1);

    int x=max(0,blockNum-converted_h+1);
    int startrow=rangeup+(x*s);
    int startcol=converted_w*(blockNum-x);

    while(startrow<rangedn && startcol>=0)
    {
        for(int m=0;m<s;m++)
        {
            if(tid<=r+m && tid>=l+m)
            {
                int row=startrow+m;
                int col=startcol+(tid-l-m);

                B[(row)*converted_h*converted_w + col]=arr[tid];
                __syncthreads;
            }
        }
        startrow+=s;
        startcol-=converted_w;
    }
   

}

int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
        int height,width,channel,size_kernel;
        printf("Enter values of image height,width,number of channels and kernel size:");
        scanf("%d %d %d %d",&height,&width,&channel,&size_kernel);
       
        int numElements = height*width*channel;

        int converted_h=height-size_kernel+1;
        int converted_w=width-size_kernel+1;

        int converted_numElements = converted_h*converted_w*channel*size_kernel*size_kernel;

        size_t size = numElements * sizeof(float);
        size_t converted_size = converted_numElements * sizeof(float);

        //printf("[Vector addition of %d elements]\n", numElements);

        // Allocate the host input vector A
        float *h_A = (float *)malloc(size);

        // Allocate the host input vector B
        float *h_B = (float *)malloc(converted_size);


        // Verify that allocations succeeded
        if (h_A == NULL || h_B == NULL)
        {
            fprintf(stderr, "Failed to allocate host vectors!\n");
            exit(EXIT_FAILURE);
        }

        // Initialize the host input vectors
        for (int i = 0; i < numElements; ++i)
        {
            h_A[i]=i;
        }
       
        // Allocate the device input vector A
        float *d_A = NULL;
        err = cudaMalloc((void **)&d_A, size);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Allocate the device input vector B
        float *d_B = NULL;
        err = cudaMalloc((void **)&d_B, converted_size);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

       
        // Copy the host input vectors A and B in host memory to the device input vectors in
        // device memory
       // printf("Copy input data from the host memory to the CUDA device\n");
        err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }


        // Launch the Vector Add CUDA Kernel
        dim3 block(width,1,1);
        dim3 grid(1,height,channel);
        //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);4
 
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        Im2Col_optimised<<<grid, block>>>(d_A, d_B,height,width,channel,size_kernel);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("The elapsed time in gpu was %f ms", milliseconds);

        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch Im2Col_optimised kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy the device result vector in device memory to the host result vector
        // in host memory.
       // printf("Copy output data from the CUDA device to the host memory\n");
        err = cudaMemcpy(h_B, d_B, converted_size, cudaMemcpyDeviceToHost);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector B from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }



        // Verify that the result vector is correct
        float A[height][width][channel];
        float check[size_kernel*size_kernel*channel][converted_h*converted_w];

/*
        float **check=(float **)malloc(converted_h*sizeof(float *));
        for(int i=0;i<converted_h;i++)
            check[i]=(float *)malloc(converted_w*sizeof(float));
*/
        for(int gid=0;gid<numElements;gid++)
        {
            int k=gid/(height*width);
            int i=(gid%(height*width))/width;
            int j=(gid%(height*width))%width;
            A[i][j][k]=h_A[gid];
        }
 
        clock_t cpu_start, cpu_end;
        double cpu_time_used;
     
        cpu_start = clock();

        for(int i=0;i+size_kernel<=height;i++)
        {
            for(int j=0;j+size_kernel<=width;j++)
            {
                for(int k=0;k<channel;k++)
                {
                    int row=k*size_kernel*size_kernel;
                    int col=i*(width-size_kernel+1) + j;
                    int cnt=0;

                    for(int l=i;l<i+size_kernel;l++)
                    {
                        for(int m=j;m<j+size_kernel;m++)
                        {
                            check[row+cnt][col]=A[l][m][k];
                            cnt++;
                        }
                    }
                }
            }
        }
 
        cpu_end = clock();
        cpu_time_used = ((double) (cpu_end - cpu_start)) / CLOCKS_PER_SEC;
        printf("\nTime elapsed in serial execution:%f ms\n",cpu_time_used*1000.00);

        int gid=0,flag=1;
        for(int i=0;i<size_kernel*size_kernel*channel;i++)
        {
            for(int j=0;j<converted_h*converted_w;j++)
            {
                if(check[i][j]!=h_B[gid])
                    flag=0;
                gid++;
            }
        }

        if(flag)
            printf("Success!!\n");
        else
            printf("Failure!!\n");






    //  for(int i=0;i<converted_numElements;i++){
      //  printf("%lf ",h_B[i]);
      // }


       // printf("Test PASSED\n");

        // Free device global memory
        err = cudaFree(d_A);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        err = cudaFree(d_B);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

       
        // Reset the device and exit
        // cudaDeviceReset causes the driver to clean up all state. While
        // not mandatory in normal operation, it is good practice.  It is also
        // needed to ensure correct operation when the application is being
        // profiled. Calling cudaDeviceReset causes all profile data to be
        // flushed before the application exits
        err = cudaDeviceReset();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
       
        printf("\n");
        // Free host memory
        free(h_A);
        free(h_B);

        //printf("Done\n");
   
    return 0;
}