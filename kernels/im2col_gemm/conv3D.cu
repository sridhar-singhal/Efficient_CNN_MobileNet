#include "../diagonal_refactorization/depth_conv.cuh"
#include "../pointwise_Conv/pointwise_conv.cuh"

void conv3d(float* d_input_mat, float* d_wt_mat, float** out_mat, int height, int width, int channels,  int op_channels, int K, int stride)
{

    cudaError_t error = cudaSuccess;

    if (d_wt_mat == NULL)
    {
        fprintf(stderr, "conv3d: No Kernel Paramaters Provided\n");
    }

    if(d_input_mat == NULL)
    {
        fprintf(stderr, "conv3d: Input Matrix memory not allocated\n");
        exit(EXIT_FAILURE);       
    }
    printf("width = %d\n", width);
    printf("height = %d\n", height);

    int width_col = ceil(float((width- K))/stride) + 1;
    int height_col = ceil(float((height - K))/stride) + 1;
    size_t totalThreads = channels*K*K*height_col*width_col;

    printf("op_width = %d\n", width_col);
    printf("op_height = %d\n", height_col);

    float* d_out_mat = NULL;    


    float* d_col = NULL;

    error = cudaMalloc((void **)&d_col, totalThreads*sizeof(float));
    if(error != cudaSuccess) {
        fprintf(stderr,"conv3d: cudaMalloc for d_col %s\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }


    dim3 gridDim(ceil((height_col*width_col)/32.0), channels, 1);
    dim3 blockDim(32, K*K, 1);

    im2col<<<gridDim ,blockDim>>>(d_input_mat, d_col, K, channels, height, width, height_col, width_col, stride);

    //A, B, height_A, width_A, height_B, Width_B
    cuBlasGemm_rowMaj(d_wt_mat,d_col, &d_out_mat,op_channels, channels*K*K ,channels*K*K, height_col*width_col);

    cudaFree(d_col);

    *out_mat = d_out_mat; //This matrix shall be returned
}


int main(int argc, char const *argv[])
{
	int height = 7, width = 7, channels = 5, op_channels = 3, stride = 2;
	int op_height = 3, op_width = 3;
	//stride is fixed to 1.
	int K = 3;

	float* h_A = NULL;
	float* h_B = NULL;
	float *d_A = NULL;
	float *d_B = NULL;
	float *d_wt = NULL;

    cudaError_t error = cudaSuccess;

	size_t size = height*width*channels*sizeof(float);

    float* wt_mat = (float *)malloc((op_channels*channels*K*K)*sizeof(float));
    if(wt_mat==NULL)
    {
    	fprintf(stderr, "Unable to allocate memory for weight matrix\n");
    	exit(EXIT_FAILURE);
    }

	h_A = (float *)malloc((channels*height*width)*sizeof(float));
    if(h_A==NULL)
    {
    	fprintf(stderr, "Unable to allocate memory for input matrix\n");
    	exit(EXIT_FAILURE);
    }   

	h_B = (float *)malloc((op_channels*op_height*op_width)*sizeof(float));
    if(h_A==NULL)
    {
    	fprintf(stderr, "Unable to allocate memory for output matrix\n");
    	exit(EXIT_FAILURE);
    }     


    for(int i = 0; i < channels*K*K*op_channels; i ++)
    {
          wt_mat[i] = 1;
    }

    for(int i = 0; i<channels*height*width;i++)
    {
    	h_A[i] = 1;
    }

    error = cudaMalloc((void **)&d_A, (channels*height*width)*sizeof(float));        
    if(error != cudaSuccess) {
        fprintf(stderr,"conv3d_example: Error in cudaMalloc for Input Matrix %s\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }


    error = cudaMalloc((void **)&d_wt, (channels*K*K*op_channels)*sizeof(float));        
    if(error != cudaSuccess) {
        fprintf(stderr,"conv3d_example: Error in cudaMalloc for weight Matrix %s\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }   


    error = cudaMemcpy(d_A, h_A, channels*height*width*sizeof(float), cudaMemcpyHostToDevice);
    if(error != cudaSuccess) {
        fprintf(stderr,"conv3d_example: Error in copying input matrix to Device %s\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    } 

    error = cudaMemcpy(d_wt, wt_mat, channels*op_channels*K*K*sizeof(float), cudaMemcpyHostToDevice);
    if(error != cudaSuccess) {
        fprintf(stderr,"conv3d_example: Error in copying weigth matrix to Device %s\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    } 

    conv3d(d_A, d_wt, &d_B, height, width, channels, op_channels, K, stride);

    error = cudaMemcpy(h_B, d_B, op_channels*op_height*op_width*sizeof(float), cudaMemcpyDeviceToHost);
    if(error != cudaSuccess) {
        fprintf(stderr,"conv3d_example: Error in copying Output Matrix to host %s\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    } 

    printf("Done\n");


    // for (int i = 0; i < op_channels; ++i)
    // {
    // 	for (int j = 0; j < op_width*op_height; ++j)
    // 	{
    // 		printf("%1.1f ", *(h_B + i*op_height*op_width + j));
    // 		if ((j+1)%op_width==0)
    // 		{
    // 			printf("\n");
    // 		}
    // 	}

    // 	printf("\n");
    // }


    free(h_A);
    free(h_B);
    free(wt_mat);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_wt);




	return 0;
}


