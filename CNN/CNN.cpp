//coding:utf-8
/*****************************************
Program: CNN
Description: 
Author: cshanbo@gmail.com
Date: 2016-08-02 10:21:52
Last modified: 2016-08-02 16:25:02
GCC version: 4.9.3
*****************************************/

#include <iostream>
#include <tuple>
#include <vector>
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

void CNN::conv2d(tensor4<double>&, tensor4<double>&, tensor4<double>&, tensor4<double>&) {

}

int main() {
    matrix<double> mt = vector<vector<double>>(2, vector<double>(3, 0));
    print(mt);
    return 0;
}
