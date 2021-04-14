#include "depth_conv.cuh"

//Linking cpp and cu is definitely difficult, it is easier to make everything in cu only.
int main(int argc, char const *argv[])
{
	int value = depth_conv_example();
	if(value == 0)
		printf("Done!!\n");
	return 0;
}