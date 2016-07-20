//coding:utf-8
/******************************************
Program: MLP cpp
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-20 13:42:10
Last modified: 2016-07-20 16:39:25
GCC version: 4.7.3
std = C++ 11
******************************************/

#include "../include/MLP.h"
#include "../include/HiddenLayer.h"
#include <cmath>

MLP::MLP() {}

MLP::MLP(int n_in, int n_out, int n_hidden, vector<vector<double>> input) {
    this->n_in = n_in;
    this->n_out = n_hidden;
    this->n_hidden = n_hidden;
    this->input = input;
    this->hiddenLayer = HiddenLayer(n_in, n_out, input, 0);
    //this->logisticLayer = LogisticRegression();
    this->logisticLayer = LogisticRegression(this->hiddenLayer.output, n_hidden, n_out);
}

MLP::~MLP() {}

double MLP::L1(vector<vector<double>>& w1, vector<vector<double>>& w2) {
    double ret = 0;
    for(auto v: w1)
        for(auto d: v)
            ret += abs(d);
    for(auto v: w2)
        for(auto d: v)
            ret += abs(d);
    return ret;
}

double MLP::L2(vector<vector<double>>& w1, vector<vector<double>>& w2) {
    double ret = 0;
    for(auto v: w1)
        for(auto d: v)
            ret += d * d;
    for(auto v: w2)
        for(auto d: v)
            ret += d * d;
    return sqrt(ret);
}

int main() {
    return 0;
}
