To run all the programs related to IM2COL
1)Save the code as im2col.cu
2)!nvcc /content/src/im2col.cu -o /content/src/im2col
3)!/content/src/im2col
4)Enter the values of height of image , width of image, number of channels and size of kernel

The code will automatically initialize a matrix with specified dimensions and work in the GPU. Also a naive implementation is done in the CPU side to check the correctness of the output.
If the output is correct it shows Success else Failure.
It will show the exectution time in both GPU and CPU.


To run all the programs related to GEMM
1)Save the code as gemm.cu
2)!nvcc /content/src/gemm.cu -o /content/src/gemm
3)!/content/src/gemm
4)Enter the values of length and width of matrix 1 and length and width of matrix 2

The code will automatically initialize a matrix with specified dimensions and work in the GPU. Also a naive implementation is done in the CPU side to check the correctness of the output.
If the output is correct it shows Success else Failure.
It will show the exectution time in both GPU and CPU.
