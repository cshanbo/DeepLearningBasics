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
    int in_dim;
    int out_dim;
    vector<vector<double>> weight;
    vector<vector<double>> input;
    vector<double> bias;

public:
    LogisticRegression();
    LogisticRegression(vector<vector<double>>, int, int);
    ~LogisticRegression();
    void train(vector<int>, vector<int>, double);
    void sigmoid(vector<double>&);
    void test(vector<int>, vector<double>);
    double negativeLogLikelihood();
};

#endif
