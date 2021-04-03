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
    
    int tid_j = blockIdx.x*blockDim.x + threadIdx.x;	//column number
    int tid_i = blockIdx.y*blockDim.y + threadIdx.y;	//row number
    int gid = tid_i*(height_col*width_col) + tid_j;    //global_id when reading row major form
    
    if(tid_j < (height_col*width_col))	
    {
        int c_im = blockIdx.y;

        int c = gid/(height_col*width_col);			//row in which we are working on in the o/p matrix 
        
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
    int gid = blockIdx.x*blockDim.x + threadIdx.x;		//global Thread Id
    if(gid < channels*K*K)
    {
      int row = gid/(K*K);  			//the row in the diagonalized weight matrix that this thread has to work on 
      int off_set = row*(K*K*channels) + row*(K*K)+gid%(K*K); //Exact position where we have to put the value
      out_wt_mat[off_set] = wt_mat[gid]; //assignment
      
    }
}

void gpuCublasMmul(cublasHandle_t handle,  float *A,  float *B, float *reference,  int m,  int k,  int n) {
    int lda=m,ldb=k,ldc=m;
    //A = m*k, B = k*n, C = m*n
    const float alf = 1;				
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
    // do the actual multiplication
    
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,alpha,B,n,A,k,beta,reference,n);    
}

void depth_conv( cublasHandle_t handle, float *mat, float *weights, float *out_mat, int stride, int channels, int K, int height, int width)
{
    int width_col = (width- K)/stride + 1;
    int height_col = (height - K)/stride + 1;
    size_t totalThreads = channels*K*K*height_col*width_col;            //total elements im2col operation
    size_t dim1 = channels*K*K;                                         //size of weight matrix
    size_t dim2 = channels*channels*K*K;                                //size of output weight matrix
    size_t size = channels*height*width;								//Total number of data points in the input image

    float* d_mat = NULL;												//input image in device memory
    cudaMalloc((void **)&d_mat, size*sizeof(float));
    cudaMemcpy(d_mat, mat, size*sizeof(float), cudaMemcpyHostToDevice);
    float* d_col = NULL;												//output after im2col in device memory
    cudaMalloc((void **)&d_col, totalThreads*sizeof(float));
    
    
    float num_th = 128.0;												//number of threads in x direction of Block for weight rearrangement 
    dim3 gridWeightDim(ceil((channels*K*K)/num_th), 1, 1);				//grid dimensions for rearrange_weights
    dim3 blockWeightDim(num_th, 1, 1);									//block dimensions for rearrange_weights
 
 	dim3 gridDim(ceil((height_col*width_col)/32.0), channels, 1);		//grid dimensions for im2col
    dim3 blockDim(32, K*K, 1);											//block dimensions for im2col
 
    float* d_wt_mat = NULL;												//weight matrix in the devo4ice memory
    cudaMalloc((void **)&d_wt_mat, dim1*sizeof(float));
    cudaMemcpy(d_wt_mat, weights, dim1*sizeof(float), cudaMemcpyHostToDevice);
	
    float* d_out_wt_mat = NULL;											//output diagonalized weight matrix in device memory
    cudaMalloc((void **)&d_out_wt_mat, dim2*sizeof(float));
    
    printf("rearrange_weights -- Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
    gridWeightDim.x, gridWeightDim.y, gridWeightDim.z, blockWeightDim.x, blockWeightDim.y, blockWeightDim.z);

    rearrange_weights<<<gridWeightDim, blockWeightDim>>>(d_wt_mat, d_out_wt_mat, K, channels);
    float* rearranged_weights = (float *)malloc(dim2*sizeof(float));	//rearranged diagonalized weights in CPU memory
    cudaMemcpy(rearranged_weights, d_out_wt_mat, dim2*sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Output rearrangement for weights :\n");
    for (int i = 0; i < (channels); ++i)
    {
        for (int j = 0; j < (channels*K*K); ++j)
        {
            printf("%1.1f ", rearranged_weights[i*(channels*K*K) + j]);
        }
        printf("\n");
    }
	
    cudaMemcpy(d_out_wt_mat, rearranged_weights, dim2*sizeof(float), cudaMemcpyHostToDevice);
    
    im2col<<<gridDim ,blockDim>>>(d_mat, d_col, K, channels, height, width, height_col, width_col, stride);

    float* col_mat = (float *)malloc(dim1*sizeof(float));
    cudaMemcpy(col_mat, d_col, (totalThreads)*sizeof(float), cudaMemcpyDeviceToHost); //copied for printing

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
 
    
    float* d_out_mat = NULL;
    cudaMalloc((void **)&d_out_mat, channels*width_col*height_col*sizeof(float));        
    
	int nr_rows_A = channels;
    int nr_cols_A = channels*K*K;
    int nr_cols_B = height_col*width_col;
    
	gpuCublasMmul(handle,d_out_wt_mat, d_col, d_out_mat, nr_rows_A, nr_cols_A, nr_cols_B); //multiply diagonalized weights with im2col 
    
	cudaMemcpy(out_mat, d_out_mat, channels*width_col*height_col*sizeof(float), cudaMemcpyDeviceToHost); //store finally in out_mat
}

int main()
{
    int K, height, width, stride, channels;	//kernel size , height of image, width of image, stride, number of channels in the input
    printf("Enter kernel size , height of image, width of image, stride, cnumber of channels in the input\n");
    
    //K SHOULD NOT BE LARGER THAN 5, NOT NEEDED IN THIS ARCHITECTURE ANYWAY. OURS IS CONSTRAINED BY BLOCK DIMENSIONS
    // 6*6*32 > 1024
 
    scanf("%d",&K);
    scanf("%d",&height);
    scanf("%d",&width);
    scanf("%d",&stride);
    scanf("%d",&channels);
    // K = 2; height = 6; width = 6; stride = 2; channels = 4;
    
    int group_size = 2;													//number of channels in a group 
    int num = ceil(channels/group_size);								//number of groups
    
    int width_col = (width- K)/stride + 1;			//effective width, that is the number of steps the kernel can be shifted along the width
    int height_col = (height - K)/stride + 1;		////effective height, that is the number of steps the kernel can be shifted along the height
    
    float* wt_mat = (float *)malloc((channels*K*K)*sizeof(float));
    for(int i = 0; i < channels*K*K; i ++)
    {
		  wt_mat[i] = i;						//initializing weight matrix with consecutive natural numbers
    }
    
    printf("Weight Matrix \n");
    for(int i = 0; i < channels; i++)
    {
    	for(int j = 0; j < K; j++)
      {
      	for(int k = 0; k < K;k++)
        {
        	printf("%1.1f ",wt_mat[i*K*K + j*K + k]);		//printing the weight matrix 
        }
        	printf("\n");
      }
      printf("\n");
    }
    
 	size_t size = channels*height*width;
    float* input_mat = (float *)malloc(size*5*sizeof(float));
    
    for(int i = 0; i < size*5; i++)
    {
    		input_mat[i] = i;						//initializing input image matrix with consecutive natural numbers
    }
 
    cublasHandle_t handle1;
    cublasCreate(&handle1);							//handle creation for using cuBLAS method in gpuCublasMmul
    
    float* out_mat = (float *)malloc(channels*height_col*width_col*sizeof(float));

    int input_offset;						//offset to be provided to input image according to group number
    int weight_offset;						//offset to be provided to weight matrix according to group number
    int output_offset;						//offset to be provided to final output matrix according to group number
    int current_channels = group_size;		//number of channels in the current group, which is different only for possibly the last group
    
    for(int i = 0; i < channels; i+= group_size)
    {
    	input_offset = height*width*i;
		weight_offset = K*K*i;
		output_offset = height_col*width_col*i;
		if ((channels - i) < group_size)
			current_channels = channels - i;
	
		depth_conv(handle1, input_mat+input_offset, wt_mat+weight_offset, out_mat+output_offset , stride, current_channels, K,  height, width);
    }
    
    printf("Printing the input image\n");
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
    }
} 