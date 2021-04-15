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
  
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate( & start);
  cudaEventCreate( & stop);
  cudaEventRecord(start);

  //L1
	init_image(&in1, l1.height, l1.width, l1.channels);
	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

  cudaEvent_t start_temp, stop_temp;
  float milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
  
	conv3d(in1,kernel,& outmat, l1.height,l1.width,l1.channels,l1.op_channels,l1.K, l1.stride);
	
  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 1 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;

  //L2
	l1.K = 3; l1.height = 112; l1.width = 112; l1. channels = 32; 
	l1.op_channels = 1; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

  milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	depth_convDriver(in1,kernel,&outmat,l1.height,l1.width,l1.stride,l1.channels,l1.K);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 2 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;

  //L3
	l1.K = 1; l1.height = 112; l1.width = 112; l1. channels = 32; 
	l1.op_channels = 64; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

  milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	pointwise_convDriver(in1,&outmat,kernel,l1.height,l1.width,l1.channels,l1.op_channels);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 3 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;
 
  //L4
  l1.K = 3; l1.height = 112; l1.width = 112; l1. channels = 64; 
	l1.op_channels = 1; l1.stride = 2;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	depth_convDriver(in1,kernel,&outmat,l1.height,l1.width,l1.stride,l1.channels,l1.K);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 4 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;

  //L5
	l1.K = 1; l1.height = 56; l1.width = 56; l1. channels = 64; 
	l1.op_channels = 128; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	pointwise_convDriver(in1,&outmat,kernel,l1.height,l1.width,l1.channels,l1.op_channels);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 5 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;
 
  //L6
  l1.K = 3; l1.height = 56; l1.width = 56; l1. channels = 128; 
	l1.op_channels = 128; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	depth_convDriver(in1,kernel,&outmat,l1.height,l1.width,l1.stride,l1.channels,l1.K);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 6 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;

  //L7
	l1.K = 1; l1.height = 56; l1.width = 56; l1. channels = 128; 
	l1.op_channels = 128; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	pointwise_convDriver(in1,&outmat,kernel,l1.height,l1.width,l1.channels,l1.op_channels);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 7 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;
 
  //L8
  l1.K = 3; l1.height = 56; l1.width = 56; l1. channels = 128; 
	l1.op_channels = 128; l1.stride = 2;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	depth_convDriver(in1,kernel,&outmat,l1.height,l1.width,l1.stride,l1.channels,l1.K);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 8 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;

  //L9
	l1.K = 1; l1.height = 28; l1.width = 28; l1. channels = 128; 
	l1.op_channels = 256; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	pointwise_convDriver(in1,&outmat,kernel,l1.height,l1.width,l1.channels,l1.op_channels);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 9 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;

  //L10
  l1.K = 3; l1.height = 28; l1.width = 28; l1. channels = 256; 
	l1.op_channels = 256; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	depth_convDriver(in1,kernel,&outmat,l1.height,l1.width,l1.stride,l1.channels,l1.K);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 10 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;

  //L11
	l1.K = 1; l1.height = 28; l1.width = 28; l1. channels = 256; 
	l1.op_channels = 256; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	pointwise_convDriver(in1,&outmat,kernel,l1.height,l1.width,l1.channels,l1.op_channels);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 11 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;
 
  //L12
  l1.K = 3; l1.height = 28; l1.width = 28; l1. channels = 256; 
	l1.op_channels = 256; l1.stride = 2;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	depth_convDriver(in1,kernel,&outmat,l1.height,l1.width,l1.stride,l1.channels,l1.K);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 12 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;

  //L13
	l1.K = 1; l1.height = 14; l1.width = 14; l1. channels = 256; 
	l1.op_channels = 512; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	pointwise_convDriver(in1,&outmat,kernel,l1.height,l1.width,l1.channels,l1.op_channels);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 13 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;
 
  //L14
  l1.K = 3; l1.height = 14; l1.width = 14; l1. channels = 512; 
	l1.op_channels = 512; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	depth_convDriver(in1,kernel,&outmat,l1.height,l1.width,l1.stride,l1.channels,l1.K);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 14 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;

  //L15
	l1.K = 1; l1.height = 14; l1.width = 14; l1. channels = 512; 
	l1.op_channels = 512; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	pointwise_convDriver(in1,&outmat,kernel,l1.height,l1.width,l1.channels,l1.op_channels);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 15 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;
 
 //L16
  l1.K = 3; l1.height = 14; l1.width = 14; l1. channels = 512; 
	l1.op_channels = 512; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	depth_convDriver(in1,kernel,&outmat,l1.height,l1.width,l1.stride,l1.channels,l1.K);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 16 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;

  //L17
	l1.K = 1; l1.height = 14; l1.width = 14; l1. channels = 512; 
	l1.op_channels = 512; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	pointwise_convDriver(in1,&outmat,kernel,l1.height,l1.width,l1.channels,l1.op_channels);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 17 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;
 
 //L18
  l1.K = 3; l1.height = 14; l1.width = 14; l1. channels = 512; 
	l1.op_channels = 512; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	depth_convDriver(in1,kernel,&outmat,l1.height,l1.width,l1.stride,l1.channels,l1.K);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 18 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;

  //L19
	l1.K = 1; l1.height = 14; l1.width = 14; l1. channels = 512; 
	l1.op_channels = 512; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	pointwise_convDriver(in1,&outmat,kernel,l1.height,l1.width,l1.channels,l1.op_channels);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 19 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;
 
 //L20
  l1.K = 3; l1.height = 14; l1.width = 14; l1. channels = 512; 
	l1.op_channels = 512; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	depth_convDriver(in1,kernel,&outmat,l1.height,l1.width,l1.stride,l1.channels,l1.K);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 20 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;

  //L21
	l1.K = 1; l1.height = 14; l1.width = 14; l1. channels = 512; 
	l1.op_channels = 512; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	pointwise_convDriver(in1,&outmat,kernel,l1.height,l1.width,l1.channels,l1.op_channels);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 21 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;
 
 //L22
  l1.K = 3; l1.height = 14; l1.width = 14; l1. channels = 512; 
	l1.op_channels = 512; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	depth_convDriver(in1,kernel,&outmat,l1.height,l1.width,l1.stride,l1.channels,l1.K);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 22 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;

  //L23
	l1.K = 1; l1.height = 14; l1.width = 14; l1. channels = 512; 
	l1.op_channels = 512; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	pointwise_convDriver(in1,&outmat,kernel,l1.height,l1.width,l1.channels,l1.op_channels);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 23 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;


  //L24
  l1.K = 3; l1.height = 14; l1.width = 14; l1. channels = 512; 
	l1.op_channels = 512; l1.stride = 2;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	depth_convDriver(in1,kernel,&outmat,l1.height,l1.width,l1.stride,l1.channels,l1.K);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 24 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;

  //L25
	l1.K = 1; l1.height = 7; l1.width = 7; l1. channels = 512; 
	l1.op_channels = 1024; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	pointwise_convDriver(in1,&outmat,kernel,l1.height,l1.width,l1.channels,l1.op_channels);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 25 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;

  //L26
  l1.K = 3; l1.height = 7; l1.width = 7; l1. channels = 1024; 
	l1.op_channels = 1024; l1.stride = 2;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	depth_convDriver(in1,kernel,&outmat,l1.height,l1.width,l1.stride,l1.channels,l1.K);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 26 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;

  //L27
	l1.K = 1; l1.height = 7; l1.width = 7; l1. channels = 1024; 
	l1.op_channels = 1024; l1.stride = 1;

	get_kernel(&kernel,l1.K,l1.channels,l1.op_channels);

	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	pointwise_convDriver(in1,&outmat,kernel,l1.height,l1.width,l1.channels,l1.op_channels);

  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 27 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;
  

  //L28
	l1.K = 7; l1.height = 7; l1.width = 7; l1. channels = 1024; 
	l1.op_channels = 1024; l1.stride = 0;



	milliseconds_temp = 0;
  cudaEventCreate(&start_temp);
  cudaEventCreate(&stop_temp);
  cudaEventRecord(start_temp);
 
	// pointwise_convDriver(in1,&outmat,kernel,l1.height,l1.width,l1.channels,l1.op_channels);
  	avgPool(in1,&outmat,l1.height, l1.width,l1.channels);
  cudaEventRecord(stop_temp);
  cudaEventSynchronize(stop_temp);
  cudaEventElapsedTime( & milliseconds_temp, start_temp, stop_temp);
  printf("The time taken to pass through layer 28 was %f ms\n", milliseconds_temp);

	cudaFree(in1);
	cudaFree(kernel);
	in1 = outmat; outmat = NULL;



  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime( & milliseconds, start, stop);
  printf("The total time taken to feed forward pass through 27 layers was %f ms\n", milliseconds);



	printf("Done!!\n");
	cudaFree(in1);
 
	return 0;
}