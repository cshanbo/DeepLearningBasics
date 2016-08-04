//coding:utf-8
/*****************************************
Program: Complete CNN 
Description: 
Author: cshanbo@gmail.com
Date: 2016-08-04 08:14:16
Last modified: 2016-08-04 10:15:17
GCC version: 4.9.3
*****************************************/

#include "../include/CNN.h"
#include "../include/utils.h"
#include "../include/HiddenLayer.h"
#include "../include/LR.h"

CNN::CNN() {}

CNN::~CNN() {}

CNN::CNN(tensor4<double>& inp, tuple<int, int, int, int> inshape, tuple<int, int, int, int> outshape, tuple<int, int> poolingShapes, int n_in, int n_h_out, int n_out) {
    this->input = inp; 
    //convolutional layer
    convLayer = ConvolutionLayer(inp, inshape, outshape, poolingShapes);
    matrix<double> h_input;
    //flatten the output of convolutional layer into 2d matrix
    flatten2(convLayer.output, h_input);
    //hidden layer
    hiddenLayer = HiddenLayer(n_in, n_h_out, h_input, 0);
    //output layer (logistic layer)
    logisticLayer = LogisticRegression(hiddenLayer.output, n_h_out, n_out);
}

int main() {
    //the update is hard to write. might need some time
    return 0;
}
