%%cuda --name gemm.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
    
 //Naive implementation of GEMM operation in GPU 

__global__ void gemm(float *A, float *B,float *C,int m,int n,int k)
{
   //Calculation of index that will be worked upon
	int i=blockIdx.y*blockDim.y+threadIdx.y;
	int j=blockIdx.x*blockDim.x+threadIdx.x;
   
	if((i<m)&&(j<n))
	{
		float val=0.0;

      //Loop to multiply and add row of A with col of B
		for(int l=0;l<k;l++)
			val+=A[l*m+i]*B[j*k+l];
		C[j*m+i]=val;
	}   
}
 
int main(void)
{
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  // Input the length and width of the matrices
  int length1,length2,width1,width2;

  printf("Enter values of length1,width1,length2,width2:");
  scanf("%d %d %d %d",&length1,&width1,&length2,&width2);

  //Calculate size of matrices in bytes
  
  int numElements = length1*width1;
  size_t size = numElements * sizeof(float);

  int numElements2 = length2*width2;
  size_t size2 = numElements2 * sizeof(float);

  int numElements3 = length1*width2;
  size_t size3 = numElements3 * sizeof(float);

  // Allocate the host input vector A
  float *h_A = (float *)malloc(size);

  // Allocate the host input vector B
  float *h_B = (float *)malloc(size2);

  // Allocate the host output vector C
  float *h_C = (float *)malloc(size3);


  // Verify that allocations succeeded
  if (h_A == NULL || h_B == NULL || h_C==NULL)
  {
      fprintf(stderr, "Failed to allocate host vectors!\n");
      exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors for testing purpose
  for (int i = 0; i < numElements; ++i)
  {
      h_A[i]=i%8;
  }
  for (int i = 0; i < numElements2; ++i)
  {
      h_B[i]=i%8;
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
  err = cudaMalloc((void **)&d_B, size2);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  // Allocate the device output vector C
  float *d_C = NULL;
  err = cudaMalloc((void **)&d_C, size3);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

 
  // Copy the host input vectors A and B in host memory to the device input vectors in device memory
  
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_B, h_B, size2, cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }


  // Launch the GEMM CUDA Kernel

  //Define block and grid dimensions
  dim3 block(32,32,1);
  dim3 grid(max(length1,length2)/32 + 1,max(width1,width2)/32 + 1,1);

  //Use cuda events to determine time taken
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  //Launch the gemm kernel
  gemm<<<grid, block>>>(d_A, d_B, d_C,length1,width2,width1);

  //Calculate the time taken by the Kernel
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("The elapsed time in gpu was %f ms\n", milliseconds);

  //Check for any error in launch of kernel
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to launch gemm kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  // Copy the device result vector in device memory to the host result vector in host memory.

  err = cudaMemcpy(h_C, d_C, size3 , cudaMemcpyDeviceToHost);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  // Verify that the result vector is correct by performing the operation in CPU

  float arr[length1][width2]; //Result array made in CPU side to verify the results

  for(int i=0;i<length1;i++){
      for(int j=0;j<width2;j++){
          arr[i][j]=0;
      }
  }


  //Monitor Time taken in serial execution in CPU side for comparison
  clock_t cpu_start, cpu_end;
  double cpu_time_used;
  cpu_start = clock();

  int f=0;                    
  for(int i=0;i<length1;i++)
  {
      for(int j=0;j<width2;j++)
      {
          for(int k=0;k<length2;k++)
          {
              arr[i][j]+=h_A[k*length1 + i]*h_B[j*length2 + k];
          }

          if(arr[i][j]!=h_C[j*length1 + i]){
              f=1;
          }
      }
  }

  /*
  Code to print both side results if necessary

  for(int i=0;i<length1;i++)
  {
      for(int j=0;j<width2;j++)
      {
          printf("%f ",arr[i][j]);
      }
      printf("\n");
  }
  for(int i=0;i<length1;i++)
  {
      for(int j=0;j<width2;j++)
      {
          printf("%f ",h_C[j*length1 + i]);
      }
      printf("\n");
  }
  */
  
  //Serial time execution printing
  cpu_end = clock();
  cpu_time_used = ((double) (cpu_end - cpu_start)) / CLOCKS_PER_SEC;
  printf("\nTime elapsed in serial execution:%f ms\n",cpu_time_used*1000.00);

  //If both CPU side and GPU side results match or not
  if(!f)
      printf("Success!!\n");
  else
      printf("Failure!!\n");        

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

  err = cudaFree(d_C);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  
  // Reset the device and exit
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
  free(h_C);
 
  return 0; 
}
 

