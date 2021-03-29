%%cuda --name im2col.cu

#include <stdio.h>

#include <stdlib.h>

#include <cuda.h>

#include <cuda_runtime.h>

#include <time.h>


__global__ void Im2Col_unoptimised(float * A, float * B, int h, int w, int c, int s) {
    //A - input matrix(h*w*c)
    //size of kernel - s
    //B - output matrix
    
    //computing the required dimenions
    int blockNum = blockIdx.z * (gridDim.y * gridDim.x) + blockIdx.x * gridDim.y + blockIdx.y;
    int threadNum = threadIdx.z * (blockDim.y * blockDim.x) + threadIdx.x * (blockDim.y) + threadIdx.y;
    int globalThreadId = blockNum * (blockDim.y * blockDim.x * blockDim.z) + threadNum;

    //starting point of kernel moves from [0][0] to [converted_h-1][converted_w-1]
    int converted_h = h - s + 1;
    int converted_w = w - s + 1;

    int k = globalThreadId / (h * w);    //channel number
    int i = (globalThreadId % (h * w)) / w;   //row number
    int j = (globalThreadId % (h * w)) % w;   //column number

    if (k < c && j <= (w - s) && i <= (h - s)) {
        //if the kernel starting point is valid
        int cnt = 0;
        for (int l = i; l < i + s; l++) {
            for (int m = j; m < j + s; m++) {
                //iterating through a kernel

                //computing the row and col in the output matrix where the element needs to be stored
                int row = k * s * s;
                int col = i * (w - s + 1) + j;

                B[(row + cnt) * converted_h * converted_w + col] = A[globalThreadId + (m - j) + ((l - i) * w)];
                cnt = cnt + 1;
            }
        }
    }

}

int main(void) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Take the image and kernel size as input
    int height, width, channel, size_kernel;
    printf("Enter values of image height, width and number of channels: ");
    scanf("%d %d %d", & height, & width, & channel);
    printf("Enter value of kernel size: ");
    scanf("%d", & size_kernel);
    
    //image dimensions
    int numElements = height * width * channel;

    //starting point of kernel moves from [0][0] to [converted_h-1][converted_w-1]
    int converted_h = height - size_kernel + 1;
    int converted_w = width - size_kernel + 1;

    //total number of elements in the final output matrix
    int converted_numElements = converted_h * converted_w * channel * size_kernel * size_kernel;

    size_t size = numElements * sizeof(float);
    size_t converted_size = converted_numElements * sizeof(float);

    // Allocate the host input vector A
    float * h_A = (float * ) malloc(size);

    // Allocate the host output vector B
    float * h_B = (float * ) malloc(converted_size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vector
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = i;
    }

    // Allocate the device input vector A
    float * d_A = NULL;
    err = cudaMalloc((void ** ) & d_A, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector B
    float * d_B = NULL;
    err = cudaMalloc((void ** ) & d_B, converted_size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vector A in host memory to the device input vectors in device memory
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //grid and block dimensions
    dim3 block(32, 8, 4);
    dim3 grid(4, 32, 8);

    //to record the time consumed for im2col operation
    cudaEvent_t start, stop;
    cudaEventCreate( & start);
    cudaEventCreate( & stop);
    cudaEventRecord(start);

    // Launch the Im2Col CUDA Kernel
    Im2Col_unoptimised << < grid, block >>> (d_A, d_B, height, width, channel, size_kernel);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime( & milliseconds, start, stop);

    printf("The elapsed time in gpu was %f ms", milliseconds);

    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch Im2Col_optimised kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device output vector in device memory to the host output vector in host memory.
    err = cudaMemcpy(h_B, d_B, converted_size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector B from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }



    // Verify that the result vector is correct
    float A[height][width][channel];
    float check[size_kernel * size_kernel * channel][converted_h * converted_w];

    //creating the actual 3-D image from the 1-D array
    for (int gid = 0; gid < numElements; gid++) {
        int k = gid / (height * width);
        int i = (gid % (height * width)) / width;
        int j = (gid % (height * width)) % width;
        A[i][j][k] = h_A[gid];
    }

    clock_t cpu_start, cpu_end;
    double cpu_time_used;

    cpu_start = clock();

    //creating the output matrix
    for (int i = 0; i + size_kernel <= height; i++) {
        for (int j = 0; j + size_kernel <= width; j++) {
            for (int k = 0; k < channel; k++) {
                int row = k * size_kernel * size_kernel;
                int col = i * (width - size_kernel + 1) + j;
                int cnt = 0;

                for (int l = i; l < i + size_kernel; l++) {
                    for (int m = j; m < j + size_kernel; m++) {
                        check[row + cnt][col] = A[l][m][k];
                        cnt++;
                    }
                }
            }
        }
    }

    cpu_end = clock();
    cpu_time_used = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    printf("\nTime elapsed in serial execution:%f ms\n", cpu_time_used * 1000.00);

    //checking whether the output matrix cretaed in CPU and GPU are same
    int gid = 0, flag = 1;
    for (int i = 0; i < size_kernel * size_kernel * channel; i++) {
        for (int j = 0; j < converted_h * converted_w; j++) {
            if (check[i][j] != h_B[gid])
                flag = 0;
            gid++;
        }
    }

    if (flag) //if the two matrix are same
        printf("Success!!\n");
    else
        printf("Failure!!\n");


    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Reset the device and exit
    err = cudaDeviceReset();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("\n");
    // Free host memory
    free(h_A);
    free(h_B);

    return 0;
}