void init_image(float** in1, int height, int width, int channels)
{
	float *h_A;
	float *d_A;
	time_t t;
	srand((unsigned) time(&t));

	cudaError_t error = cudaSuccess;

	h_A = (float *)malloc((channels*height*width)*sizeof(float));
    if(h_A==NULL)
    {
    	fprintf(stderr, "init_image: Unable to allocate memory for input matrix\n");
    	exit(EXIT_FAILURE);
    }  
    
    error = cudaMalloc((void **)&d_A, (channels*height*width)*sizeof(float));        
    if(error != cudaSuccess) {
        fprintf(stderr,"init_image: Error in cudaMalloc for Input Matrix %s\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < channels*height*width; ++i)
    {
    	h_A[i] = rand()%100;
    }

    error = cudaMemcpy(d_A, h_A, channels*height*width*sizeof(float), cudaMemcpyHostToDevice);
    if(error != cudaSuccess) {
        fprintf(stderr,"init_image: Error in copying input matrix to Device %s\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    *in1 = d_A;
    free(h_A);     
}

void get_kernel(float** d_wt, int K, int channels, int op_channels)
{
	float *h_A;
	float *d_A;
	time_t t;
	srand((unsigned) time(&t));
	cudaError_t error = cudaSuccess;

  int size = K*K*channels*op_channels;
	h_A = (float *)malloc(size*sizeof(float));
    if(h_A==NULL)
    {
    	fprintf(stderr, "init_image: Unable to allocate memory for input matrix\n");
    	exit(EXIT_FAILURE);
    }  
    
    error = cudaMalloc((void **)&d_A, size*sizeof(float));        
    if(error != cudaSuccess) {
        fprintf(stderr,"init_image: Error in cudaMalloc for Input Matrix %s\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < size; ++i)
    {
    	h_A[i] = rand()%100;
    }

    error = cudaMemcpy(d_A, h_A, size*sizeof(float), cudaMemcpyHostToDevice);
    if(error != cudaSuccess) {
        fprintf(stderr,"init_image: Error in copying input matrix to Device %s\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    *d_wt = d_A;     
}

struct shape
{
	int K;
	int channels;
	int height;
	int width;
	int op_channels;
	int stride;
};

//int depth_convDriver(float* d_input_mat, float* d_wt_mat, float ** out_mat, int height, int width, int stride, int channels, int K)
int forward_pass()
{
	struct shape l1;
	l1.K = 3; l1.height = 224; l1.width = 224; l1. channels = 3; l1.op_channels = 32;
	l1.stride = 2;
	float* in1 = NULL;
	float* kernel = NULL;
	float* outmat = NULL;

	init_image(&in1, l1.height, l1.width, l1.channels);
	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	conv3d(in1,kernel,& outmat, l1.height,l1.width,l1.channels,l1.op_channels,l1.K, l1.stride);
	
	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;

	l1.K = 3; l1.height = 112; l1.width = 112; l1. channels = 32; 
	l1.op_channels = 1; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	depth_convDriver(in1,kernel,&outmat,l1.height,l1.width,l1.stride,l1.channels,l1.K);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;

	l1.K = 1; l1.height = 112; l1.width = 112; l1. channels = 32; 
	l1.op_channels = 64; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	pointwise_convDriver(in1,&outmat,kernel,l1.height,l1.width,l1.channels,l1.op_channels);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;


	printf("Done!!\n");
	cudaFree(in1);

	return 0;
}