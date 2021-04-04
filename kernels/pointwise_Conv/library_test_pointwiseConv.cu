#include "pointwise_conv.cuh"
//Command to compile:
//nvcc -lcublas -o op library_test_pointwiseConv.cu 


int main(int argc, char const *argv[])
{
	int value = pointwise_conv_example();
	if(value == 0)
		printf("Done!!\n");
	return 0;
}