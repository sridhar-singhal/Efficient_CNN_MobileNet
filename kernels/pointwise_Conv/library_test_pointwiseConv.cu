#include "pointwise_conv.cuh"
//Command to compile:
//nvcc -lcublas -o op library_test_pointwiseConv.cu 


int main(int argc, char const *argv[])
{	
	int height, width, channels, op_channels;
	op_channels = 64;
	channels = 64;
	height = width = 32;
	int verification = 1; //0 implies no verification
	int value = pointwise_conv_example(height,width,channels,op_channels,verification);
	if(value == 0)
		printf("Done!!\n");
	return 0;
}