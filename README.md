# Efficient_CNN_MobileNet
Efficient CNN Implementation of Mobile Net Architectures using Diagonalization

Guidlines:

Create Working Directories for each person. Remember to pull before you push. Alsp remember to commit to your local branch before you pull else you could lose all preogress. Else all the collective code could be potentially wiped out. I would recommend creating branches and working on them. Also, work and commit to your own branches and do not touch the main. 

Work only in your own folder, it is a temporary place for you to work. Later everything shall be shifted to final_library with integration. 


## CNN Inferencing
The **classification.cpp** file serves as the complete end to end input image to output classification script. Ensure that you have caffe installed in your machine and the system paths are appropriately appended for the inclusion of caffe libraries in the script. 
- The script takes as input the **prototxt file**, the **caffemodel file**, the **mean file**, and the **label file**.
- The prototxt file is provided in the same folder. 
- The model file containing the input weights needs to be downloaded from the official [MobileNet repo](https://github.com/shicai/MobileNet-Caffe). 

The **convolution_separable.cu** implements the depthwise separable convolutional layers as is utilized in the MobileNet architecture.

The **eval.cpp** is a standalone script which can be utlized for image preprocessing before the image is passed through any convolutional neural network architecture.

The **parse.cpp** is a parsing mechanism for prototxt scripts appropriately converted to text files through the help of any json parser.


## Misc Links

Links
https://developer.nvidia.com/blog/unified-memory-cuda-beginners/ : Unified Memory, can be accessed by both CPU and GPU. This might mean we can load the data here first as the CPU would not allow loading very large matrices.

https://stackoverflow.com/questions/13202492/maximum-size-of-malloc/13202597 : Finding largest possible memory allocation

https://www.researchgate.net/post/What-is-the-maximum-size-of-an-array-in-C : If declared as a local variable, it is stored in stack of the program, which is terribly small. If used Malloc, still can not use more memory than allocated memory for the program. Even though for a 32 bit system we can get 4 GB and for 64 bit system we can get exabytes of memory, not enough space is actually present. 

https://dl.acm.org/doi/fullHtml/10.1145/3280851 : Memory access Coalesing
https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/#:~:text=Misaligned%20Data%20Accesses&text=Arrays%20allocated%20in%20device%20memory,are%20aligned%20to%20their%20size. : Another one on memory access Coalescing

