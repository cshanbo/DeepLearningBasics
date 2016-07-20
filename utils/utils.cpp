//coding:utf-8
/******************************************
Program: utils cpp
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-20 13:13:40
Last modified: 2016-07-20 13:19:22
GCC version: 4.7.3
std = C++ 11
******************************************/

#include "../include/utils.h"
#include <iostream>
#include <cstdlib>
#include <assert.h>

double randRange(double fMin,  double fMax) {
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

vector<vector<double>> dot(vector<vector<double>>& input, vector<vector<double>>& matrix, vector<double> bias) {
    assert(!(input.empty() || matrix.empty()));
    assert(!(input[0].size() != matrix.size() || (bias.size() != 0 && matrix[0].size() != bias.size())));

    vector<vector<double>> ret(input.size(), vector<double>(matrix[0].size(), 0));
    for(unsigned int i = 0; i < input.size(); ++i)
        for(unsigned int j = 0; j < matrix[0].size(); ++j)
            for(unsigned int k = 0; k < input[0].size(); ++k)
                ret[i][j] += input[i][k] * matrix[k][j];

    if(bias.size() != 0)
        for(unsigned int i = 0; i < ret.size(); ++i)
            for(unsigned int j = 0; j < ret[0].size(); j++)
                ret[i][j] += bias[j];
    return ret;
}
