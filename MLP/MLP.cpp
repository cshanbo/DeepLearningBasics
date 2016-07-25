//coding:utf-8
/******************************************
Program: MLP cpp
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-20 13:42:10
Last modified: 2016-07-24 20:43:05
GCC version: 4.7.3
std = C++ 11
******************************************/

#include "../include/MLP.h"
#include "../include/HiddenLayer.h"
#include "../include/utils.h"

MLP::MLP() {}

MLP::~MLP() {}

MLP::MLP(int n_in, int n_out, int n_hidden, vector<vector<double>> input) {
    this->n_in = n_in;
    this->n_out = n_hidden;
    this->n_hidden = n_hidden;
    this->input = input;
    this->hiddenLayer = HiddenLayer(n_in, n_out, input, 0);
    this->logisticLayer = LogisticRegression(this->hiddenLayer.output, n_hidden, n_out);
}

double MLP::l1_norm(vector<vector<double>>& w1, vector<vector<double>>& w2) {
    return L1(w1) + L2(w2);
}

double MLP::l2_norm(vector<vector<double>>& w1, vector<vector<double>>& w2) {
    return L2(w1) + L2(w2);
}

double MLP::cost(vector<int> y, double l1_rate, double l2_rate) {
    return logisticLayer.negativeLogLikelihood(y) + l1_rate * l1_norm(hiddenLayer.weights, logisticLayer.weights) + l2_rate * l2_norm(hiddenLayer.weights, logisticLayer.weights);
}

//int main() {
//    return 0;
//}
