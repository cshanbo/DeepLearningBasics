//coding:utf-8
/*
Program: Hidden Layer.cpp
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-20 09:27:14
Last modified: 2016-07-25 09:56:50
GCC version: 4.7.3
*/

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <string>
#include "../include/HiddenLayer.h"
#include "../include/utils.h"
//gradient(tanh(x)) == 1 - tanh(x) ** 2

using namespace std;

HiddenLayer::HiddenLayer() {}

HiddenLayer::~HiddenLayer() {}

HiddenLayer::HiddenLayer(int n_in, int n_out, vector<vector<double>> input, int activationType) {
    //Initialization. this kind of initialization for weights has been proved to be good
    this->n_in = n_in;
    this->n_out = n_out;
    this->input = input;
    weights = vector<vector<double>>(n_in, vector<double>(n_out, 0));
    //initial bias
    bias = vector<double>(n_out, 0);
    //initial weight
    for(unsigned int i = 0; i < weights.size(); ++i)
        for(unsigned int j = 0; j < weights[0].size(); ++j)
            weights[i][j] = randRange(-1 * sqrt(6.0 / (n_in + n_out)), sqrt(6.0 / (n_in + n_out)));
    dot(input, weights, output, bias);
    activation(output, activationType);
}

void HiddenLayer::update(double rate, vector<int> y) {
    assert(!y.empty());
    double dy = 0;
    for(unsigned int k = 0; k < input.size(); ++k) {
        for(int i = 0; i < n_out; ++i) {
            for(int j = 0; j < n_in; ++j) {
                dy = (1 - tanh(output[k][i])) * input[k][j]* (y[k] == i? 1 - output[k][i]: -1 * output[k][i]);
                weights[j][i] += rate * dy / input.size();
                bias[i] += rate * dy / input.size();
            }
        }
    }
    dot(input, weights, output, bias);
    activation(output, 0);
}

void HiddenLayer::activation(vector<double>& vec, int s) {
    //sigmoid is a little different
    //0: tanh, 1: sigmoid, 2: ReLU
    if(s == 0)
        for(unsigned int i = 0; i < vec.size(); ++i)
            vec[i] = tanh(vec[i]);
    else if(s == 1)
        for(unsigned int i = 0; i < vec.size(); ++i)
            vec[i] = 4.0 / (1 + exp(-1 * vec[i]));
    else if(s == 2)
        for(unsigned int i = 0; i < vec.size(); ++i)
            vec[i] = vec[i] >= 0? vec[i]: 0;
}

void HiddenLayer::activation(vector<vector<double>>& vec, int s) {
    //sigmoid is a little different
    //0: tanh, 1: sigmoid, 2: ReLU
    if(s == 0)
        for(unsigned int i = 0; i < vec.size(); ++i)
            for(unsigned int j = 0; j < vec[0].size(); ++j)
                vec[i][j] = tanh(vec[i][j]);

    else if(s == 1)
        for(unsigned int i = 0; i < vec.size(); ++i)
            for(unsigned int j = 0; j < vec[0].size(); ++j)
                vec[i][j] = 4.0 / (1 + exp(-1 * vec[i][j]));

    else if(s == 2)
        for(unsigned int i = 0; i < vec.size(); ++i)
            for(unsigned int j = 0; j < vec[0].size(); ++j)
                vec[i][j] = vec[i][j] >= 0? vec[i][j]: 0;
}

int main() {
    vector<vector<double>> input{
        {1, 1, 1, 0, 0, 0},
        {1, 0, 1, 0, 0, 0},
        {1, 1, 1, 0, 0, 0},
        {0, 0, 1, 1, 1, 0},
        {0, 0, 1, 1, 0, 0},
        {0, 0, 1, 1, 1, 0}, 
        {0, 0, 0, 0, 0, 1}, 
    };

    vector<int> ytrain{0, 0, 0, 1, 1, 1, 1};

    vector<vector<double>> test{
        {1, 0, 1, 0, 0, 0},
        {0, 0, 1, 1, 1, 0}, 
        {0, 0, 0, 1, 1, 1}, 
        {0, 1, 0, 0, 0, 0}, 
        {0, 1, 1, 1, 1, 1}, 
        {0, 0, 0, 1, 0, 1}, 
    };

    vector<int> ytest{0, 1, 1, 0, 1, 1};

    HiddenLayer hidden(6, 2, input, 0);
    for(int i = 0; i < 50; ++i)
        hidden.update(0.01, ytrain);

    print(hidden.weights);

    vector<int> label;
    vector<vector<double>> ygx;
    dot(test, hidden.weights, ygx, hidden.bias);
    hidden.activation(ygx, 0);
    for(auto v: ygx)
        label.push_back(maxIndex(v));
    double precision = 0;
    for(unsigned int i = 0; i < ytest.size(); ++i) {
        precision = ytest[i] != label[i] ? precision: precision + 1;
        cout << ytest[i] << '\t' << label[i] << endl;
    }
    precision /= ytest.size();
    cout << "Precision:" << precision << endl;

    return 0;
}
