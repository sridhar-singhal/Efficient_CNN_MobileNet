//Library for pointwise convolution

#include "../im2col_gemm/im2col_optimised.cuh"
#include <cublas_v2.h>
int cuBlasGemm_rowMaj(float* A, float* B, float** op, int height_A, int width_A, int height_B, int width_B)
{	
	//C = AB is implemented
	float* C = NULL;

	cudaError_t err = cudaSuccess;
	cublasStatus_t  stat;
	cublasHandle_t handle;
	
	if(A == NULL || B == NULL)
	{
		fprintf(stderr, "cuBlasGemm_rowMaj: Input Matrix not Provided\n");
		exit(EXIT_FAILURE);
	}

	if(width_A != height_B)
	{
		fprintf(stderr, "cuBlasGemm_rowMaj: Input matrix size mismatch\n");
	}

	const float alpha = 1.0f, beta = 0.0f;

	stat = cublasCreate(&handle);
		if (stat != CUBLAS_STATUS_SUCCESS) {
	    fprintf(stderr, "!!!! CUBLAS initialization error\n");
	    return EXIT_FAILURE;
	}

	err = cudaMalloc((void**) &C, height_A*width_B*sizeof(float));
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "cuBlasGemm_rowMaj: Failed to allocate device array for Output (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, width_B,height_A,height_B, &alpha, B,width_B, A,height_B,&beta, C,width_B);

	*op = C;
	return 0;
	

}

int pointwise_convDriver(float* A, float** m_B, float* d_kernels, int height, int width, int channels, int op_channels)
{
	float* im2col_mat = NULL;
	int success = Im2Col_driver(A,&im2col_mat,height,width,channels,1);
	if (im2col_mat == NULL)
	{
		fprintf(stderr, "pointwise_covDriver: Input matrix not returned after Im2Col\n");
		exit(EXIT_FAILURE);
	}

	float* op = NULL;
	cuBlasGemm_rowMaj(d_kernels, im2col_mat,&op,op_channels,channels,channels,height*width);
	*m_B = op;
	cudaFree(im2col_mat); 
	return 0;
}

//Since the kernels are of small size, we have used device memory only. 
//For larger kernels, kernel would have to be launched for each layer as the each filter would be of larger size.
//m_B is returned as a row major matrix, each row denotes one fully opened 1X1Xchannels matrix
//https://arxiv.org/pdf/1803.09926.pdf Page 2 : Fig 2
int pointwise_kernelsToDevice(float** kernels, float** m_B, int channels, int op_channels)
{
	cudaError_t err = cudaSuccess;
	if (kernels == NULL)
	{
		fprintf(stderr, "Error: No Pointwise Conv Kernel Paramaters Provided\n");
	}

	float* h_B = NULL;
    h_B = (float * ) malloc(channels*op_channels*sizeof(float));
    if (h_B == NULL)
    {
    	fprintf(stderr, "pointwise_kernelsToDevice: Failed To allocate memory of Host Array\n");
    }
    for (int i = 0; i < op_channels; ++i)
    {
    	for (int j = 0; j < channels; ++j)
    	{
	    	*(h_B + i*channels + j) = *(*(kernels + i) + j); 
    	}
    }

	float* d_B = NULL;
	int size = channels*op_channels*sizeof(float);
	err = cudaMalloc((void**)&d_B,size);

    if (err != cudaSuccess) 
    {
        fprintf(stderr, "Failed to allocate device array for Pointwise Conv Paramaters (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector h_B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }   

    *m_B = d_B;
    free(h_B);
    return 0;
}	


int pointwise_conv_example()
{	
	float *d_A = NULL, *d_B = NULL, *h_A = NULL, *h_B = NULL, *h_C = NULL;

	cudaError_t err = cudaSuccess;	//Error Codes for CUDA
	int success = 0;
	
	int height,width,channels,op_channels;

    printf("Enter values of image height, width, channels :");
    scanf("%d %d %d", & height, & width, & channels);
    printf("Enter value of output channels: ");
    scanf("%d", & op_channels);	
    
    //image dimensions
    int numElements = height * width * channels;

    //starting point of kernel moves from [0][0] to [converted_h-1][converted_w-1]
    // int converted_h = height - size_kernel + 1;
    // int converted_w = width - size_kernel + 1;

    //total number of elements in the final output matrix
    int converted_numElements = height*width*op_channels;

    size_t size = numElements * sizeof(float);
    size_t converted_size = converted_numElements * sizeof(float);

    // Allocate the host input vector A
	h_A = (float * ) malloc(size);

    // Allocate the host output vector B
    h_B = (float * ) malloc(converted_size);

    h_C = (float* ) malloc(converted_size);

    if (h_A == NULL || h_B == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vector
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = i;
    }

    //Allocate memory and copy host vector to device
    err = cudaMalloc((void ** ) & d_A, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Generate the Kernel Matrices
    //It is assumed that each 1X1Xchannels size of block is available.
    //op_channels number of suck blocks exist.
    float** kernels = NULL;
    float* temp = NULL;
    kernels = (float**)malloc(op_channels*sizeof(float*));
    if (kernels == NULL)
    {
    	fprintf(stderr, "Unable to allocate memory location for pointer to Kernels\n");
    }
    for (int i = 0; i < op_channels; ++i)
    {
    	temp = (float* )malloc(channels*sizeof(float));
    	if(temp == NULL)
    	{
    		fprintf(stderr, "Unable to allocate memory for Kernel numer : %d\n",i+1);
    	}
    	for (int j = 0; j < channels; ++j)
    	{
    		*(temp + j) = j+1;
    	}
    	*(kernels + i) = temp;
    	temp = NULL;
    }


    float* d_kernels = NULL;
    success = pointwise_kernelsToDevice(kernels,&d_kernels,channels,op_channels);

    if(d_kernels == NULL)
    {
    	fprintf(stderr, "GPU does not contain Kernels\n");
    	exit(EXIT_FAILURE);
    }

    success = pointwise_convDriver(d_A,&d_B, d_kernels, height,width,channels,op_channels);



    err = cudaMemcpy(h_B,d_B,converted_size,cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) 
    {
	    fprintf(stderr, "Failed to copy vector B from device to host (error code %s)!\n", cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
    }
    printf("%d\n", success);

    //_________Verification________________

    int sum_element = 0, temp_value = 0;;
    for (int l = 0; l < op_channels; ++l)
    {
    	for (int i = 0; i < height; ++i)
    	{
    		for (int j = 0; j < width; ++j)
    		{
    			sum_element = 0;
    			for (int k = 0; k < channels; ++k)
    			{	
    				temp_value = *(*(kernels + l) + k);
    				sum_element += h_A[k*height*width + i*width + j]* temp_value;
    			}
    			h_C[l*height*width + i*width + j] = sum_element;
    		}
    	}
    }

    for (int i = 0; i < converted_numElements; ++i)
    {
    	if(h_B[i] != h_C[i])
    	{
    		printf("PointWise Convolution Failed: CPU Not equal GPU!!");
    		exit(EXIT_FAILURE);
    	}
    }

    printf("PointWise Convolution success!!\n");

    free(h_A);
    free(h_B);
    free(h_C);
    for (int i = 0; i < op_channels; ++i)
    {
    	temp = *(kernels + i);
    	free(temp);
    }

    free(kernels);

    cudaFree(d_A);
    cudaFree(d_kernels);
    cudaFree(d_B);







	return 0;
}
