// %%cuda --name depth_conv.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))


__global__ void im2col(float *mat, float *col, int K, int channels, int height, int width, int height_col, int width_col, int stride)
{
    
    int tid_j = blockIdx.x*blockDim.x + threadIdx.x;    //column number
    int tid_i = blockIdx.y*blockDim.y + threadIdx.y;    //row number
    int gid = tid_i*(height_col*width_col) + tid_j;    //global_id when reading row major form
    
    if(tid_j < (height_col*width_col))
    {
        int c_im = blockIdx.y;

        int c = gid/(height_col*width_col);//row in which we are working on in the o/p matrix 
        
        int h_offset = (c/K)%K;
        int w_offset = c%K;
        int h =  (gid%(height_col*width_col))/width_col;
        int w = gid%width_col;


        
        int h_pad = h*stride + h_offset;
        int w_pad = w*stride + w_offset;
        
        int index = (c_im * height + h_pad) * width + w_pad;
        
        col[gid] = mat[index];
            
    }
}


__global__ void rearrange_weights(float* wt_mat, float* out_wt_mat, int K, int channels)
{
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < channels*K*K)
    {
      int row = gid/(K*K);  //the row in the final output matrix that this thread has to work on 
      int off_set = row*(K*K*channels) + row*(K*K) + gid%(K*K); //Exact position where we have to put the value
      out_wt_mat[off_set] = wt_mat[gid];
      
    }
}


void gpuCublasMmul(float *A,  float *B, float *reference,  int m,  int k,  int n) {
    //int lda=m,ldb=k,ldc=m;
    //A = m*k, B = k*n, C = m*n
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
    // do the actual multiplication
    
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,alpha,B,n,A,k,beta,reference,n);    
}

void depth_conv(float *d_mat, float * d_wt_mat, float **out_mat, int stride, int channels, int K, int height, int width, float* im2col_time, float* diag_time, float* cublas_time)
{
    int width_col = (width- K)/stride + 1;
    int height_col = (height - K)/stride + 1;
    size_t totalThreads = channels*K*K*height_col*width_col;            //total elements im2col operation
    size_t dim1 = channels*K*K;                                         //size of weight matrix
    size_t dim2 = channels*channels*K*K;                                //size of output weight matrix
    size_t size = channels*height*width;

    //mat is copied to d_mat and d_mat is used ahead.
    cudaError_t error = cudaSuccess;
 

    if(d_mat == NULL)
    {
        fprintf(stderr, "depth_convDriver: Input Matrix memory not allocated\n");
        exit(EXIT_FAILURE);       
    }

    // float* d_mat = NULL;
    // error = cudaMalloc((void **)&d_mat, size*sizeof(float));
    // if(error != cudaSuccess) {
    //     fprintf(stderr,"Some Error in cudaMalloc for d_mat %s\n",cudaGetErrorString(error));
    //     exit(EXIT_FAILURE);
    // }
 
    // cudaMemcpy(d_mat, mat, size*sizeof(float), cudaMemcpyHostToDevice);
    
    float* d_col = NULL;

    error = cudaMalloc((void **)&d_col, totalThreads*sizeof(float));
    if(error != cudaSuccess) {
        fprintf(stderr,"depth_convDriver: cudaMalloc for d_col %s\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    
    cudaDeviceProp devp;
    cudaGetDeviceProperties(&devp, 0);
    // printf("Warp Size: %d\n", devp.warpSize);
    // printf("Max number of threads per block: %d\n", devp.maxThreadsPerBlock);

    float num_th = 128.0;
    dim3 gridWeightDim(ceil((channels*K*K)/num_th), 1, 1);
    dim3 blockWeightDim(num_th, 1, 1);
 
      dim3 gridDim(ceil((height_col*width_col)/32.0), channels, 1);
    dim3 blockDim(32, K*K, 1);
 
    float* d_wt_mat = NULL;
    error = cudaMalloc((void **)&d_wt_mat, dim1*sizeof(float));
    if(error != cudaSuccess) {
        fprintf(stderr,"depth_convDriver: Some Error in cudaMalloc for d_wt_mat %s\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    
    //weights are in a single matrix, row major. 
    //Assumed to be available in system host RAM. Changed to device
    // cudaMemcpy(d_wt_mat, weights, dim1*sizeof(float), cudaMemcpyHostToDevice);
    if (d_wt_mat == NULL)
    {
        fprintf(stderr, "depth_convDriver: No Kernel Paramaters Provided\n");
    }
    float* d_out_wt_mat = NULL;
    //float* out_wt_mat = (float *)calloc(dim2, sizeof(float));
    error = cudaMalloc((void **)&d_out_wt_mat, dim2*sizeof(float));
    if(error != cudaSuccess) {
        fprintf(stderr,"depth_convDriver: Some Error in cudaMalloc for d_out_wt_mat %s\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    
    //cudaMemcpy(d_out_wt_mat, out_wt_mat, dim2*sizeof(float), cudaMemcpyHostToDevice);

    //printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
    //gridWeightDim.x, gridWeightDim.y, gridWeightDim.z, blockWeightDim.x, blockWeightDim.y, blockWeightDim.z);

    cudaEvent_t start3, stop3;
    float milliseconds3 = 0;
    cudaEventCreate( & start3);
    cudaEventCreate( & stop3);
    cudaEventRecord(start3);

    rearrange_weights<<<gridWeightDim, blockWeightDim>>>(d_wt_mat, d_out_wt_mat, K, channels);

    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);
    cudaEventElapsedTime( & milliseconds3, start3, stop3);
    printf("Weight Diagonalization: The elapsed time in GPU was %f ms\n", milliseconds3);
    *diag_time = milliseconds3;


    //float* rearranged_weights = (float *)malloc(dim2*sizeof(float));
    //cudaMemcpy(rearranged_weights, d_out_wt_mat, dim2*sizeof(float), cudaMemcpyDeviceToHost);
    
    //printf("Output rearrangement:\n");
    /*for (int i = 0; i < (channels); ++i)
    {
        for (int j = 0; j < (channels*K*K); ++j)
        {
            printf("%1.1f ", rearranged_weights[i*(channels*K*K) + j]);
        }
        printf("\n");
    }*/
    //cudaMemcpy(d_out_wt_mat, rearranged_weights, dim2*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEvent_t start1, stop1;
    float milliseconds1 = 0;
    cudaEventCreate( & start1);
    cudaEventCreate( & stop1);
    cudaEventRecord(start1);

    im2col<<<gridDim ,blockDim>>>(d_mat, d_col, K, channels, height, width, height_col, width_col, stride);
    
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime( & milliseconds1, start1, stop1);
    printf("Im2Col : The elapsed time in GPU was %f ms\n", milliseconds1);
    *im2col_time = milliseconds1;
    /*float* col_mat = (float *)malloc(totalThreads*sizeof(float));
    cudaMemcpy(col_mat, d_col, (totalThreads)*sizeof(float), cudaMemcpyDeviceToHost);

    printf("Printing after im2col operation\n");
    for (int i = 0; i < (channels*K*K); ++i)
    {
      for (int j = 0; j < (height_col*width_col); ++j)
      {
        printf("%1.1f ",col_mat[i*(height_col*width_col) + j]);
      }
      printf("\n");
    }
 
    cudaMemcpy(d_col, col_mat, totalThreads*sizeof(float), cudaMemcpyHostToDevice);
    */
    //printf("0\n");
    float* d_out_mat = NULL;
    error = cudaMalloc((void **)&d_out_mat, channels*width_col*height_col*sizeof(float));        
    if(error != cudaSuccess) {
        fprintf(stderr,"depth_convDriver: Error in cudaMalloc for Output Matrix (d_out_mat) %s\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    
    //printf("2\n");
    int nr_rows_A = channels;
    int nr_cols_A = channels*K*K;
    int nr_cols_B = height_col*width_col;
    
    cudaEvent_t start2, stop2;
    float milliseconds2 = 0;
    cudaEventCreate( & start2);
    cudaEventCreate( & stop2);
    cudaEventRecord(start2);


    gpuCublasMmul(d_out_wt_mat, d_col, d_out_mat, nr_rows_A, nr_cols_A, nr_cols_B);

    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime( & milliseconds2, start2, stop2);
    printf("cuBLAS : The elapsed time in GPU was %f ms\n", milliseconds2);
    *cublas_time = milliseconds2;

    // cudaMemcpy(out_mat, d_out_mat, channels*width_col*height_col*sizeof(float), cudaMemcpyDeviceToHost);
    
    // cudaFree(d_out_mat);
    // cudaFree(d_mat); //It is the duty of the input kernel to free the memory.
    cudaFree(d_col);
    // cudaFree(d_wt_mat);
    cudaFree(d_out_wt_mat);
    // cudaFree(d_out_mat);
    *out_mat = d_out_mat;
    //d_out_mat is the output matrix allocated in the memory
    
    //free(rearranged_weights);
    //free(col_mat);
}

int main()
{
    int K, height, width, stride, channels; //kernel size , height of image, width of image, stride, number of channels in the input
    printf("Enter kernel size , height of image, width of image, stride, number of channels in the input\n");
    
    //K SHOULD NOT BE LARGER THAN 5, NOT NEEDED IN THIS ARCHITECTURE ANYWAY. OURS IS CONSTRAINED BY BLOCK DIMENSIONS
    // 6*6*32 > 1024
    cudaError_t error = cudaSuccess;

    float im2_col_total = 0;
    float diag_total = 0;
    float cublas_total = 0;
 
    float im2col_time = 0;
    float diag_time = 0;
    float cublas_time = 0;

    scanf("%d",&K);
    scanf("%d",&height);
    scanf("%d",&width);
    scanf("%d",&stride);
    scanf("%d",&channels);
    /* height = 50; width = 50; channels = 12;
    stride = 1;channels = 8;K = 3;height = 1024; width = 1024; */
    
    int group_size = 8;//number of channels in a group 
    //int num = ceil(channels/group_size);
    //printf("num = %d\n",num);
    int width_col = (width- K)/stride + 1;
    int height_col = (height - K)/stride + 1;
    
    float* wt_mat = (float *)malloc((channels*K*K)*sizeof(float));
    for(int i = 0; i < channels*K*K; i ++)
    {
          wt_mat[i] = 1;
    }
    
    /*
    printf("Weight Matrix \n");
    for(int i = 0; i < channels; i++)
    {
        for(int j = 0; j < K; j++)
      {
        for(int k = 0; k < K;k++)
        {
            printf("%1.1f ",wt_mat[i*K*K + j*K + k]);
        }
            printf("\n");
      }
      printf("\n");
    }
    */
 
      size_t size = channels*height*width;
    float* input_mat = (float *)malloc(size*sizeof(float));
    
    float* d_input_mat = NULL;
    error = cudaMalloc((void **)&d_input_mat, size*sizeof(float));        
    if(error != cudaSuccess) {
        fprintf(stderr,"depth_convExample: Error in cudaMalloc for Input Matrix %s\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < size; i++)
    {
            input_mat[i] = 1;
    }
 
    error = cudaMemcpy(d_input_mat, input_mat, size*sizeof(float), cudaMemcpyHostToDevice);
    if(error != cudaSuccess) {
        fprintf(stderr,"depth_convExample: Error in copying input matrix to Device %s\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }


    
    float* out_mat = (float *)malloc(channels*height_col*width_col*sizeof(float));
    float* d_out_mat = NULL;    

    //depth_conv(handle1, input_mat, wt_mat, out_mat, stride, channels, K,  height, width);
    
    int input_offset;
    int weight_offset;
    int output_offset;
    int current_channels = group_size;
    
    cudaEvent_t start, stop;
    float milliseconds = 0;

    cudaEventCreate( & start);
    cudaEventCreate( & stop);

    cudaEventRecord(start);
    printf("Entering\n");

    for(int i = 0; i < channels; i+= group_size)
    {
        input_offset = height*width*i;
      weight_offset = K*K*i;
      output_offset = height_col*width_col*i;
      if ((channels - i) < group_size)
        current_channels = channels - i;
      
      depth_conv(input_mat+input_offset, wt_mat+weight_offset, out_mat+output_offset , stride, current_channels, K,  height, width, &im2col_time, &diag_time, &cublas_time);
      im2_col_total += im2col_time;
      diag_total += diag_time;
      cublas_total += cublas_time;
    }
    printf("Left\n");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( & milliseconds, start, stop);


    printf("Depthwise Conv: The elapsed time in GPU was %f ms\n", milliseconds);
    printf("Im2Col: The elapsed time in GPU was %f ms\n", im2_col_total);
    printf("Diagonalwise: The elapsed time in GPU was %f ms\n", diag_total);
    printf("CuBlas GEMM: The elapsed time in GPU was %f ms\n", cublas_total);    

    /*printf("Printing the input image\n");
    for (int i = 0; i < channels; ++i)
    {
      for (int j = 0; j < height; ++j)
      {
        for (int k = 0; k < width; ++k)
        {
          printf("%1.1f ",input_mat[i*height*width + j*width + k]);
        }
        printf("\n");
      }
      printf("\n");
    }
    
    printf("Printing the output matrix\n");
    for(int i = 0; i < channels; i++)
    {
        for(int j = 0; j < height_col*width_col; j++)
      {
        printf("%1.1f ", out_mat[i*height_col*width_col + j]);
      }
      printf("\n");
    }*/
    

    free(wt_mat);
    free(input_mat);
    free(out_mat);
    
}