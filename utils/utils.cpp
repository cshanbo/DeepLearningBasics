//coding:utf-8
/******************************************
Program: utils cpp
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-20 13:13:40
Last modified: 2016-07-21 13:13:17
GCC version: 4.7.3
std = C++ 11
******************************************/

#include "../include/utils.h"
#include <iostream>
#include <cstdlib>
#include <assert.h>
#include <math.h>

double randRange(double fMin,  double fMax) {
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(x));
}

int maxIndex(vector<double>& vec) {
    if(vec.empty())
        return -1;
    int ret = 0;
    double m = 0;
    for(unsigned int i = 0; i < vec.size(); ++i)
        if(vec[i] > m) {
            m = vec[i];
            ret = i;
        }
    return ret;
}

void dot(vector<vector<double>>& input, vector<vector<double>>& matrix, vector<vector<double>>& ret, vector<double> bias) {
    assert(!(input.empty() || matrix.empty()));
    assert(!(input[0].size() != matrix.size() || (bias.size() != 0 && matrix[0].size() != bias.size())));

    ret = vector<vector<double>>(input.size(), vector<double>(matrix[0].size(), 0));
    for(unsigned int i = 0; i < input.size(); ++i)
        for(unsigned int j = 0; j < matrix[0].size(); ++j)
            for(unsigned int k = 0; k < input[0].size(); ++k)
                ret[i][j] += input[i][k] * matrix[k][j];

    if(bias.size() != 0)
        for(unsigned int i = 0; i < ret.size(); ++i)
            for(unsigned int j = 0; j < ret[0].size(); j++)
                ret[i][j] += bias[j];
    return;
}

void print(vector<vector<double>> vec) {
    if(vec.empty())
        cout << "empty" << endl;
    for(auto v: vec) {
        for(auto d: v)
          cout << d << " "; 
        cout << endl;
    }
}
