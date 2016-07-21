//coding:utf-8
/*
Program: Logistic Regression CPP
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-19 16:11:48
Last modified: 2016-07-21 16:46:02
GCC version: 4.9.3
*/

#include <iostream>
#include <stdlib.h>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <math.h>
#include "../include/LR.h"
#include "../include/utils.h"
#include <vector>
using namespace std;

LogisticRegression::LogisticRegression() {}

LogisticRegression::LogisticRegression(vector<vector<double>> input, int in_d, int out_d) {
    //init weights as all 0
    n_in = in_d;
    n_out = out_d;
    this->input = input;
    weight = vector<vector<double>>(n_in, vector<double>(n_out, 0));
    bias = vector<double>(n_out, 0);
    dot(input, weight, y_given_x, bias);
    softmax(y_given_x);
    y_pred = vector<int>(input.size(), 0);
    for(unsigned int i = 0; i < y_pred.size(); ++i)
        y_pred[i] = maxIndex(y_given_x[i]);
}

LogisticRegression::~LogisticRegression() {}

void LogisticRegression::sigmoid(vector<vector<double>>& vec) {
    if(vec.empty())
        return;
    for(unsigned int i = 0; i < vec.size(); ++i)
        for (unsigned int j = 0; j < vec[0].size(); ++j)
            vec[i][j] = 1.0 / (1 + exp(-1 * vec[i][j]));
}

void LogisticRegression::softmax(vector<vector<double>>& vec) {
    if(vec.empty())
        return;
    for(unsigned int i = 0; i < vec.size(); ++i) {
        double sum = 0;
        for(unsigned int j = 0; j < vec[0].size(); ++j)
            sum += exp(vec[i][j]);
        for(unsigned int j = 0; j < vec[0].size(); ++j)
            vec[i][j] = exp(vec[i][j]) / sum;
    }
}

double LogisticRegression::negativeLogLikelihood(vector<int> y) {
    if(y.empty())
        return 0;
    double sum = 0;
    for(unsigned int i = 0; i < y.size(); ++i)
        sum += log(y_given_x[i][y[i]]);
    return -1 * sum / y.size();
}

double LogisticRegression::calcError(vector<int> y) {
    assert(y.size() == y_pred.size());
    double ret = 0;
    for(unsigned int i = 0; i < y.size(); ++i)
        ret = y[i] == y_pred[i]? ret: ret + 1.0;
    return ret / y.size();
}

void LogisticRegression::update(double rate, vector<int> y) {
    assert(!y.empty());
    double dy = 0;
    for(unsigned int k = 0; k < input.size(); ++k) {
        for(int i = 0; i < n_out; ++i) {
            dy = y[k] == i? 1 - y_given_x[k][i]: -1 * y_given_x[k][i];
            for(int j = 0; j < n_in; ++j)
                weight[j][i] += rate * dy * input[k][j] / input.size();
            bias[i] += rate * dy / input.size();
        }
    }
}


int main() {
    vector<vector<double>> input{{1, 2, 3}, {0, 1, 3}};
    vector<int> y{1, 1};
    LogisticRegression lr(input, 3, 2);
    cout << lr.negativeLogLikelihood(y) << endl;
    return 0;
}












//
