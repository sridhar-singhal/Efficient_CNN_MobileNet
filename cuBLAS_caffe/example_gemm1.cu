//Example for using GEMM
#include <stdio.h>
#include <stdlib.h>
//#include <iostream.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

//Matrix is N X M
#define M 3
#define N 3

//#define ij2l(i,j,K) (i*K + j)	
//This used to save in row major format. We need to save in coloumn major format
#define ij2l(i,j,k) (i + j*k)
//This uses coloum major format for saving matrices
//As the im2col implemented in part 3 would be row major, we would have to write our own im2col_colmaj
int main(int argc, char const *argv[])
{
	int i, j;
	float *a = NULL,*b = NULL, *c = NULL;
	//Note: float* a,b,c; does not work, each pointer has to be defined separately
	float *devA, *devB, *devC;	//Pointers in DEVICE

	//CUDA status
	cudaError_t cudaStat;
	cublasStatus_t  stat;
	cublasHandle_t handle;

	a = (float *)malloc(N*M*sizeof(float));	//N*M matrix
	b = (float *)malloc(N*N*sizeof(float));	//N*N matrix
	c = (float *)malloc(N*M*sizeof(float));	//N*M matrix output
	//Shal be doing b*a, (GEMM)
/*	for (i = 0; i < N; ++i)
	{
		for(j = 0; j<M; ++j)
		{
			*(a + ij2l(i,j,N)) = j + 1;
		}
	}*/

	//Since accesses coalesced when done consecutively, it would be better if we put the coloumn access "j" on the outer loop as j multiplies with N.
	for(j = 0; j<M ; j++)
	{
		for(i = 0;i<N;i++)
		{
			*(a + ij2l(i,j,N)) = j + 1;	//If the older implemetation of ij2l was used, "N" would be "M"
		}
	} 
	for (i = 0; i < N; ++i)
	{
		for (j = 0; j<N; ++j)
		{
			if(i == j)
			{
				*(b + ij2l(i,j,N)) = 0;
			}
			else
			{
				*(b + ij2l(i,j,N)) = 1;
			}
		}
	}
	printf("--------A matrix-----------\n");
	for (i = 0; i < N; ++i)
	{
		for (j = 0; j<M; ++j)
		{
			printf("%f ",*(a + ij2l(i,j,M)) );
		}
		printf("\n");
	}

	printf("--------B matrix-----------\n");
	for (i = 0; i < N; ++i)
	{
		for (j = 0; j<N; ++j)
		{
			printf("%f ",*(b + ij2l(i,j,N)) );
		}
		printf("\n");
	}
	cudaStat = cudaMalloc ((void**)&devA, M*N*sizeof(*a)); // Whatever *a is, float or int, space is allocated
	if (cudaStat!=cudaSuccess)
	{
		printf("Device Memory Allocation Failed\n");
		return EXIT_FAILURE;
	}
	cudaStat = cudaMalloc((void**)&devB, N*N*sizeof(*b));
	if (cudaStat!=cudaSuccess)
	{
		printf("Device Memory Allocation Failed\n");
		return EXIT_FAILURE;
	}
	cudaStat = cudaMalloc((void**)&devC, N*M*sizeof(float));


	stat = cublasSetMatrix(N,M,sizeof(*a),a,N,devA,N);
	stat = cublasSetMatrix(N,N,sizeof(*b),b,N,devB,N);


	const float alpha = 1.0f, beta = 0.0f;
	//For some reason, const float alpha = 1f, beta 0f; does not work. .0 has to be added.

	stat = cublasCreate(&handle);
		if (stat != CUBLAS_STATUS_SUCCESS) {
	    fprintf(stderr, "!!!! CUBLAS initialization error\n");
	    return EXIT_FAILURE;
	}

	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,N,&alpha,devB,N,devA,N,&beta,devC,N);	//Answer is copied in C
	//CUBLAS_OP_N implies that it is not transposed, it is neccessary to give this, else doesn't work
	//alpha and beta have to be given in as pointers, use &alpha, &beta;
	stat = cublasGetMatrix(N,M,sizeof(*c),devC,N,c,N);
	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);
	cublasDestroy(handle);

	//GEMM does alpha*A*B + beta*C. Now A is N * M, B is N * N. SO we would do B*A.

	printf("--------C matrix-----------\n");
	for (i = 0; i < N; ++i)
	{
		for (j = 0; j<M; ++j)
		{
			printf("%f ",*(c + i + j*N) ); //Coloumn First access
		}
		printf("\n");
	}

	free(a);
	free(b);
	free(c);

	return EXIT_SUCCESS;
}

//With help from https://github.com/zchee/cuda-sample/blob/master/0_Simple/matrixMulCUBLAS/matrixMulCUBLAS.cpp