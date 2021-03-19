Working Directory for Part 5 of the problem statment, cuBLASS and Caffe Pooling Kernel Operation.

im2col implementatin in Cuda and on CPU.
https://github.com/piojanu/CUDA-im2col-conv/blob/master/im2col.cu

im2col + GEMM is sometimes called as CONVGEMM where im2col is done on the fly, in the same kernel launch along with GEMM.
https://arxiv.org/pdf/2005.06410.pdf : High Performance and Portable Convolution Operators for ARM-based Multicore Processors 
They introduce the concept of convgemm and also tells about implementation of im2col.

Introduction to Caffe and Pooling
http://adilmoujahid.com/posts/2016/06/introduction-deep-learning-python-caffe/

Further Pooling can be done in a number of ways, max, min, avg, Stochastic etc.
It is a way of reducing the size of input, where we take a 2D window from the input and then output one value per window, it could be max or min or average etc as per the type of polling used.
https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8#:~:text=A%20pooling%20layer%20is%20another,in%20pooling%20is%20max%20pooling
More on convolutions and pooling layers

Caffe's pooling layer
https://caffe.berkeleyvision.org/tutorial/layers/pooling.html
Links are provided for implementations in CUDA GPU and CPU both. 


GEMM and CAFFE convolution
https://medium.com/nodeflux/demystifying-convolution-in-popular-deep-learning-framework-caffe-c74a58fe6bf8

GEMM and Convolution
https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/




MISC
Verify that CUDA is installed
https://stackoverflow.com/questions/52337791/verify-that-cublas-is-installed
Examples which were compiled, just try to go through them. I found one named simpleCUBLAS.
It is in /samples/7_CUDALibraries/simpleCUBLAS

If does not run, try to figure out using this.
https://forums.developer.nvidia.com/t/cublas-for-10-1-is-missing/71015/10


CUBLAS Functions:
https://docs.nvidia.com/cuda/cublas/index.html
http://developer.download.nvidia.com/compute/cuda/1_0/CUBLAS_Library_1.0.pdf

NOTE:
1. CUBLAS is present in /usr/local/cuda-10.2/include
with the name cublas.h
2. Thrown an error that cublas_V2 could not be found, related to that. 
https://stackoverflow.com/questions/64113574/undefined-reference-to-cublascreate-v2-in-tmp-tmpxft-0000120b-0000000-10-my
3. codes using cuBLAS functions can be combined my linking libraries properly. 
nvcc example1.cu -lcublas -o op
This works like a charm. https://docs.nvidia.com/cuda/cublas/#static-library Check out that link for more support.


