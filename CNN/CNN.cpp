//coding:utf-8
/*****************************************
Program: CNN
Description: 
Author: cshanbo@gmail.com
Date: 2016-08-02 10:21:52
Last modified: 2016-08-03 19:52:14
GCC version: 4.9.3
*****************************************/

#include <iostream>
#include <tuple>
#include <vector>
#include <utility>
#include <cassert>
#include <cmath>

#include "../include/LR.h"
#include "../include/utils.h"
#include "../include/CNN.h"

using namespace std;

CNN::CNN() {}

CNN::~CNN() {}

CNN::CNN(tensor4<double> input, tuple<int, int, int, int> inshape, tuple<int, int, int, int> fshape, tuple<int, int> pshape) {
    assert(get<1>(inshape) == get<1>(fshape));
    this->input = input;
    this->pool_size = pshape;
    this->input_shape = inshape;
    this->filter_shape = fshape;

    this->fan_in = get<1>(filter_shape) * get<2>(filter_shape) * get<3>(filter_shape);
    this->fan_out = get<0>(filter_shape) * get<2>(filter_shape) * get<3>(filter_shape) / get<0>(pool_size) / get<1> (pool_size);

    double w_bound = sqrt(6.0 / (fan_in + fan_out));
    this->weights = tensor4<double> (get<0>(filter_shape), tensor3<double>(get<1>(filter_shape), matrix<double>(get<2>(filter_shape), vector<double>(get<3>(filter_shape), 0))));
    for(auto& a: this->weights)
        for(auto& b: a)
            for(auto& c: b)
                for(auto& d: c)
                    d = randRange(-1 * w_bound, w_bound);
    
    this->bias = vector<double>(get<0>(filter_shape), 0);
}

void CNN::conv2d(tensor4<double>& input, tensor4<double>& weights, tuple<int, int, int, int>& filter_shape, tuple<int, int, int, int>& input_shape, tensor4<double>& output, bool full_conv) {
    //the pair parameter is a little bit unfriend, because pair->first is the x-axis while people usually traverse a matrix by row (y-axis, pair.second) first. This could be handled in following updates
    //
    assert(input.size() == get<0>(input_shape) && input[0].size() == get<1>(input_shape) && input[0][0].size() == get<2>(input_shape) && input[0][0][0].size() == get<1>(input_shape));
    assert(weights.size() == get<0>(filter_shape) && weights[0].size() == get<1>(filter_shape) && weights[0][0].size() == get<2>(filter_shape) && weights[0][0][0].size() == get<1>(filter_shape));

    int fm_row = get<2>(filter_shape);
    int fm_col = get<3>(filter_shape);
    if(output.empty())
        output = tensor4<double>(input.size(), tensor3<double>(get<0>(filter_shape), matrix<double>(fm_row, vector<double>(fm_col, 0))));

    // j is the jth output feature map. size is the 0th dim of weights
    // k is the kth input image of a batch. size is the 0th dim of input
    // i is the ith input feature map (channel). size is the 1th dim of input
    // each image has input[0].size() channels
    // so the weights should work on each of them
    // row is the row of an input feature map of input
    // col is the col of an input feature map of input 
    // each step, using a dot function to generate the value
    // take the average as the result
    if(full_conv)
        for(unsigned int j = 0; j < weights.size(); ++j)
            for(unsigned int k = 0; k < input.size(); ++k)
                for(unsigned int i = 0; i < input[0].size(); ++i)
                    //i is the ith input feature map 
                    //input[k][i] is a image matrix
                        // the (k, i) image
                    for(unsigned int row = 0; row < input[k][i].size(); ++row)
                        for(unsigned int col = 0; col < input[k][i][0].size(); ++col) {
                            pair<int, int> p1 = make_pair<int, int>(j, i);
                            pair<int, int> p2 = make_pair<int, int>(j + fm_col - 1, i + col - 1);
                            double temp = dotElement(input[k][i], weights[j][i], p1, p2);
                            output[k][j][row][col] += temp / k;
                        }
    else {
        //if ignoring the border
        //TODO
    }
    return;
}

void CNN::poolOut(tensor4<double>& conv_out, tensor4<double>& output, pair<int, int> pool_size, bool ignore_border) {
    //pool the output using max pooling
    //ignore the border
    int prow = pool_size.first, pcol = pool_size.second;
    output = tensor4<double>(conv_out.size(), tensor3<double>(conv_out[0].size(), matrix<double>(conv_out[0][0].size() / prow, vector<double>(conv_out[0][0][0].size() / pcol, 0))));
    if(ignore_border) {
        for(unsigned int m = 0; m < output.size(); ++m)
            for(unsigned int n = 0; n < output[0].size(); ++n)
                for(unsigned int i = 0; i < output[0][0].size(); ++i)
                    for(unsigned int j = 0; j < output[0][0][0].size(); ++j)
                        output[m][n][i][j] = maxPooling(conv_out[m][n], make_pair<int, int>(j * pool_size.second, i * pool_size.first), make_pair<int, int>(j * pool_size.second + pcol, i * pool_size.first + prow));
    } else {
        //TODO
    }
}

int main() {
    matrix<double> input = {{1, 2, 3, 4}, {2, 3, 4, 1}, {3, 1, 2, 0}};
    matrix<double> w = {
        {2, 0, 5}, 
        {1, 2, 1}, 
    };
    matrix<double> ret;
    dot(input, w, ret, make_pair<int, int>(1, 0), make_pair<int, int>(2, 1));
    print(ret);
    return 0;
}
