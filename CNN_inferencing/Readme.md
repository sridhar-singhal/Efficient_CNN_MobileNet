# CNN Inferencing
The **classification.cpp** file serves as the complete end to end input image to output classification script. Ensure that you have caffe installed in your machine and the system paths are appropriately appended for the inclusion of caffe libraries in the script. 
- The script takes as input the **prototxt file**, the **caffemodel file**, the **mean file**, and the **label file**.
- The prototxt file is provided in the same folder. 
- The model file containing the input weights needs to be downloaded from the official [MobileNet repo](https://github.com/shicai/MobileNet-Caffe). 

<br/>

The **convolution_separable.cu** implements the depthwise separable convolutional layers as is utilized in the MobileNet architecture.

<br/>

The **eval.cpp** is a standalone script which can be utlized for image preprocessing before the image is passed through any convolutional neural network architecture.

<br/>

The **parse.cpp** is a parsing mechanism for prototxt scripts appropriately converted to text files through the help of any json parser. 
- To use the parsing mechanism, process the prototxt file and rewrite the convolutional layers as `kernel_name output_size file_size pad stride` with one layer in every line. 
- The parse mechanism has been written for a single layer but can be easily extended to multiple layers. 
- The script is to be used as a parser for calling the user-defined kernels in accordance with the specified model architecture.
