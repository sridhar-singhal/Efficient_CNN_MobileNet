This has 1 super function called **depth_conv** which makes use of 3 functions: 1.) **rearrange_weights** 2.) **im2col** 3.) **gpuCublasMmul**

Theh input to the **depth_conv** function is the number of channels, stride, size of kernel, height and width of the image, the image and weight matrix.
Note that the image and weight matrix themselves are fed by the main method. 
Currently the image and weight matrix is being initialized by consecutive natural numbers. 
The **im2col** function implements a write-coalesced image to column implementation.

![Im2Col(1)](https://user-images.githubusercontent.com/50399493/112749303-2c618680-8fdf-11eb-8f20-69e89a833741.png)

The **rearrange_weights** function is used to diagonilize the weight matrix to enable multiplication of corresponding weight matrices to corresponding channels in the image:

![Weight_diagonilzation](https://user-images.githubusercontent.com/50399493/112749466-f4a70e80-8fdf-11eb-813f-040a4d85da24.png)

The **gpuCublasMmul** uses the cuBLAS operation of General Matrix Multiplication
https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication

Also note that the channels are being grouped in groups of fixed size _S_. This is done because the diagnolization increases the matrix dimensions of the 
weights significantly. Beyond a certain size matrix multiplication is slowed down by this increased size weight matrix. Thus we launch serially multiple
kernels to complete the convolution.
