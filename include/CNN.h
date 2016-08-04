//coding:utf-8
/*****************************************
Program: complete CNN header
Description: 
Author: cshanbo@gmail.com
Date: 2016-08-04 08:14:47
Last modified: 2016-08-04 08:14:47
GCC version: 4.9.3
*****************************************/

#ifndef _CNN_H_
#define _CNN_H_

#include "../include/utils.h"
#include "../include/LR.h"
#include "../include/HiddenLayer.h"
#include "../include/convolutionLayer.h"

#include <vector>
#include <tuple>

using namespace std;

class CNN {
public:
    tensor4<double> input;
    HiddenLayer hiddenLayer;
    LogisticRegression logisticLayer;
    ConvolutionLayer convLayer;
    
    CNN();
    ~CNN();
    
    //param0: input, is a 4d tensor with batch_size, n_feature_map, image_height, image_weight
    //param1: the shape of param0
    //param2: the shape of second convolution layer
    //param3: n_in for hiddenlayer
    //param4: n_out for hiddenlayer and n_in for lr layer (they are of the same size)
    //param5: n_out for lr layer
    CNN(tensor4<double>&, tuple<int, int, int, int>, tuple<int, int, int, int>, tuple<int, int>, int, int, int);

};

#endif
