//coding:utf-8
/*
Program: Logistic Regression header
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-19 16:03:33
Last modified: 2016-07-19 16:03:33
GCC version: 4.7.3
*/

#ifndef _LOGISTIC_REGRESSION_H_
#define _LOGISTIC_REGRESSION_H_

#include <vector>
using namespace std;

class LogisticRegression {
    int N;
    int in_dim;
    int out_dim;
    vector<vector<double>> weight;
    vector<double> bias;
public:
    LogisticRegression(int, int, int);
    ~LogisticRegression();
    void train(vector<int>, vector<int>, double);
    void softmax(vector<double>&);
    void test(vector<int>, vector<double>);
};

#endif
