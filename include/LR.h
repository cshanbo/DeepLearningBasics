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
public:
    int n_in;
    int n_out;
    vector<int> y_pred;
    vector<vector<double>> weights;
    vector<vector<double>> input;
    vector<double> bias;
    vector<vector<double>> y_given_x;

    LogisticRegression();
    LogisticRegression(vector<vector<double>>, int, int);
    ~LogisticRegression();

    void update(double, vector<int>);
    void sigmoid(vector<vector<double>>&);
    void softmax(vector<vector<double>>&);
    double negativeLogLikelihood(vector<int>);
    double calcError(vector<int>);

    void train(vector<vector<double>>, vector<int> y, int, double = 0.1);
    vector<int> test(vector<vector<double>>, vector<int> y);
};

#endif
