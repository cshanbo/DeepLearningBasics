//coding:utf-8
/*
Program: Hidden Layer.cpp
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-20 09:27:14
Last modified: 2016-07-20 10:44:36
GCC version: 4.7.3
*/

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <string>
#include "../include/HiddenLayer.h"
#include "../include/utils.h"

using namespace std;

HiddenLayer::HiddenLayer(int n_in, int n_out, vector<vector<double>> input) {
    //Initialization. this kind of initialization for weights has been proved to be good
    this->n_in = n_in;
    this->n_out = n_out;
    this->input = input;
    weights = vector<vector<double>>(n_in, vector<double>(n_out, 0));
    bias = vector<double>(n_out, 0);
    for(unsigned int i = 0; i < weights.size(); ++i)
        for(unsigned int j = 0; j < weights[0].size(); ++j)
            weights[i][j] = randRange(-1 * sqrt(6.0 / (n_in + n_out)), sqrt(6.0 / (n_in + n_out)));
}

HiddenLayer::~HiddenLayer() {}

void HiddenLayer::activation(vector<double>& vec, string s = "tanh") {
    if(s == "sigmoid")
        for(unsigned int i = 0; i < vec.size(); ++i)
            vec[i] = 4.0 / (1 + exp(-1 * vec[i]));
    else if(s == "tanh")
        for(unsigned int i = 0; i < vec.size(); ++i)
            vec[i] = (exp(vec[i]) - exp(-1 * vec[i])) / (exp(vec[i]) + exp(-1 * vec[i]));
    else if(s == "relu")
        for(unsigned int i = 0; i < vec.size(); ++i)
            vec[i] = vec[i] >= 0? vec[i]: 0;
}

int main() {
    cout << "test " << endl;
    return 0;
}
