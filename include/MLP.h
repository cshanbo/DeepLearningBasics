//coding:utf-8
/******************************************
Program: MLP header file
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-20 13:42:25
Last modified: 2016-07-20 13:42:25
GCC version: 4.7.3
std = C++ 11
******************************************/

#ifndef _MLP_H_
#define _MLP_H_

#include "../include/HiddenLayer.h"
#include "../include/LR.h"
class MLP {
public:
    vector<vector<double>> input;
    int n_in;
    int n_out;
    int n_hidden;	//hidden layer input dimension
    double negativeLogLikelihood;
    double errors;

    HiddenLayer hiddenLayer;
    LogisticRegression logisticLayer;    

    MLP();
    ~MLP();
    MLP(int, int, int, vector<vector<double>>); //n_in, n_out, n_hidden

    double l1_norm(vector<vector<double>>&, vector<vector<double>>&);
    double l2_norm(vector<vector<double>>&, vector<vector<double>>&);
    double cost(vector<int>, double, double);
};

#endif
