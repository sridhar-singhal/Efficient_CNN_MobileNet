// #include "kernels/diagonal_refactorization/depth_conv.cuh"
//conv3D already uses depth_conv, is creating conflict.
#include "kernels/im2col_gemm/conv3D.cuh"
// #include "kernels/pointwise_Conv/pointwise_conv.cuh"
// This is being called already in conv3D as well
#include "kernels/pooling/pooling_avg.cuh"
#include "forward/forward_pass.cuh"
//Compilation Command:
//nvcc -o op -lcublas main.cu
int main(int argc, char const *argv[])
{
	printf("Compiled?\n");
	forward_pass();
	return 0;
}