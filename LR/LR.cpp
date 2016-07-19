//coding:utf-8
/*
Program: Logistic Regression CPP
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-19 16:11:48
Last modified: 2016-07-19 17:13:32
GCC version: 4.9.3
*/

#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include "LR.h"
#include <vector>
using namespace std;

LogisticRegression::LogisticRegression(int dataSize, int in_d, int out_d) {
    //init
    N = dataSize;
    in_dim = in_d;
    out_dim = out_d;

    weight = vector<vector<double>>(out_dim, vector<double>(in_dim, 0));
    bias = vector<double>(out_dim, 0);
}

LogisticRegression::~LogisticRegression() {}

void LogisticRegression::softmax(vector<double>& vec) {
    double sum = 0;
    for(auto d: vec)
        sum += exp(d);
    for(int i = 0; i < vec.size(); i++)
        vec[i] = exp(vec[i]) / sum;
}

void LogisticRegression::train(vector<int> x, vector<int> y, double rate) {
    //x is one input vector
    //batch gradient descent
    vector<double> p_y_given_x (out_dim, 0);
    vector<double> diff(out_dim, 0);
    for(int i = 0; i < out_dim; ++i) {
        for(int j = 0; j < in_dim; ++j)
            p_y_given_x[i] += weight[i][j] * x[j];
        p_y_given_x[i] == bias[i];
    }

    softmax(p_y_given_x);
    for(int i = 0; i < out_dim; ++i) {
        diff[i] = y[i] - p_y_given_x[i];
        for(int j = 0; j < in_dim; ++j)
            weight[i][j] += rate * diff[i] * x[j] / N;
        bias[i] += rate * diff[i] / N;
    }
}

void LogisticRegression::test(vector<int> x, vector<double> y) {
    for(int i = 0; i < out_dim; ++i) {
        y[i] = 0;
        for(int j = 0; j < in_dim; ++j)
            y[i] += weight[i][j] * x[j];
        y[i] += bias[i];
    }
    softmax(y);
}

int main() {
    srand(0);
    double learning_rate = 0.1;
    int n_epochs = 500;

    int train_N = 6;
    int test_N = 2;
    int n_in = 6;
    int n_out = 2;

    vector<vector<int>> train_X = {
        {1, 1, 1, 0, 0, 0},
        {1, 0, 1, 0, 0, 0},
        {1, 1, 1, 0, 0, 0},
        {0, 0, 1, 1, 1, 0},
        {0, 0, 1, 1, 0, 0},
        {0, 0, 1, 1, 1, 0}
    };

    vector<vector<int>> train_Y = {
        {1, 0},
        {1, 0},
        {1, 0},
        {0, 1},
        {0, 1},
        {0, 1}
    };

    LogisticRegression lr(train_N, n_in, n_out);

    for(int epoch = 0; epoch < n_epochs; ++epoch)
        for(int i = 0; i < train_N; ++i)
            lr.train(train_X[i],  train_Y[i], learning_rate);

    vector<int> test_X = {
        {1, 0, 1, 0, 0, 0},
    };

    vector<double> test_Y(2);

        lr.test(test_X, test_Y);
        for(int j=0; j<n_out; j++) {
          cout << test_Y[j] << " ";
        }
        cout << endl;
    return 0;
}

