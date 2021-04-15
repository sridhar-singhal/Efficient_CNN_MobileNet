#include "conv3D.cuh"

int main(int argc, char const *argv[])
{
//int conv3d_example(int printture = 0,int height = 7, int width = 7, int channels = 5, int op_channels = 3, int stride = 2, int K = 3, int op_height = 3, int op_width = 3)

	// conv3d_example(0,7,7,3,5,2,3,3,3);
	conv3d_example();
	printf("Done\n");
	return 0;
}