//coding:utf-8
/***********************************************************
Program: Convolution Neural Network header file
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-27 10:18:29
Last modified: 2016-07-27 10:18:29
GCC version: 4.9.3
***********************************************************/

#ifndef _CONVOLUTION_LAYER_H_
#define _CONVOLUTION_LAYER_H_

#include <vector>
#include <tuple>
#include "../include/utils.h"
using namespace std;

/*
 * a 4D tensor corresponding to a mini-batch of input images. The shape of the tensor is as follows: [mini-batch size,  number of input feature maps,  image height,  image width.
 * a 4D tensor corresponding to the weight matrix W. The shape of the tensor is: [number of feature maps at layer m,  number of feature maps at layer m-1,  filter height,  filter width
 * theano.tensor.nnet.conv2d,  which is the most common one in almost all of the recent published convolutional models. In this operation,  each output feature map is connected to each input feature map by a different 2D filter,  and its value is the sum of the individual convolution of all inputs through the corresponding filter.
 * */

class ConvolutionLayer {
public:
    tensor4<double> input;

    tensor4<double> output;

    tensor4<double> weights;

    //the 4 dims of input
    tuple<int, int, int, int> input_shape;
    //the 4 dims of filter or weights
    tuple<int, int, int, int> filter_shape;
    tuple<int, int> pool_size;

    int fan_in;
    int fan_out;

    vector<double> bias;

    ConvolutionLayer();

    ~ConvolutionLayer();

    ConvolutionLayer(tensor4<double>, tuple<int, int, int, int>, tuple<int, int, int, int>, tuple<int, int> = make_tuple(2, 2));

    void conv2d(tensor4<double>&, tensor4<double>&, tuple<int, int, int, int>&, tuple<int, int, int, int>&, tensor4<double>&, bool = true);

    void poolOut(tensor4<double>&, tensor4<double>&, pair<int, int>, bool = true);

    void update(double rate, vector<double>&);
};

#endif
