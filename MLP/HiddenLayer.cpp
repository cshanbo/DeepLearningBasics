//coding:utf-8
/*
Program: Hidden Layer.cpp
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-20 09:27:14
Last modified: 2016-07-20 13:02:25
GCC version: 4.7.3
*/

#include <iostream>
#include <vector>
#include <locale>
#include <cstdlib>
#include <cmath>
#include <string>
#include "../include/HiddenLayer.h"
#include "../include/utils.h"

using namespace std;

HiddenLayer::HiddenLayer(int n_in, int n_out, vector<vector<double>> input, int activationType) {
    //Initialization. this kind of initialization for weights has been proved to be good
    this->n_in = n_in;
    this->n_out = n_out;
    this->input = input;
    weights = vector<vector<double>>(n_in, vector<double>(n_out, 0));
    bias = vector<double>(n_out, 0);
    for(unsigned int i = 0; i < weights.size(); ++i)
        for(unsigned int j = 0; j < weights[0].size(); ++j)
            weights[i][j] = randRange(-1 * sqrt(6.0 / (n_in + n_out)), sqrt(6.0 / (n_in + n_out)));
    activation(weights, activationType);
}

HiddenLayer::~HiddenLayer() {}

void HiddenLayer::activation(vector<double>& vec, int s) {
    //sigmoid is a little special
    //0: tanh, 1: sigmoid, 2: ReLU
    if(s == 0)
        for(unsigned int i = 0; i < vec.size(); ++i)
            vec[i] = (exp(vec[i]) - exp(-1 * vec[i])) / (exp(vec[i]) + exp(-1 * vec[i]));
    else if(s == 1)
        for(unsigned int i = 0; i < vec.size(); ++i)
            vec[i] = 4.0 / (1 + exp(-1 * vec[i]));
    else if(s == 2)
        for(unsigned int i = 0; i < vec.size(); ++i)
            vec[i] = vec[i] >= 0? vec[i]: 0;
}

void HiddenLayer::activation(vector<vector<double>>& vec, int s) {
    //sigmoid is a little special
    if(s == 0)
        for(unsigned int i = 0; i < vec.size(); ++i)
            for(unsigned int j = 0; j < vec[0].size(); ++j)
                vec[i][j] = (exp(vec[i][j]) - exp(-1 * vec[i][j])) / (exp(vec[i][j]) + exp(-1 * vec[i][j]));

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
    vector<vector<double>> ret;
    vector<vector<double>> input{{1, 2, 3}, {0, 1, 2}, {3, 3, 5}};
    vector<vector<double>> matrix{{1, 3}, {0, 1}, {3, 5}};
    vector<double> bias{0, 1};
    ret = dot(input, matrix);
    for(auto v: ret) {
        for(auto d: v)
            cout << d << " ";
        cout << endl;
    }
    return 0;
}
