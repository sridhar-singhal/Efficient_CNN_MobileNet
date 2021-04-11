#include "pooling_avg.cuh"

//Command to compile
//nvcc -lcublas -o op library_test_avgPool.cu 

int main(int argc, char const *argv[])
{	
	int channels = 4096*4, heigth = 3, width = 3;
	int dim[] = {3,7,11};
	int len = 3;
	int value;
	for (int i = 0; i < len; ++i)
	{
		value = avgPool_example(dim[i],dim[i],channels);
	}
	if(value == 0)
		printf("Done!!\n");
	return 0;
}
