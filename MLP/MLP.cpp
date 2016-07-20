//coding:utf-8
/******************************************
Program: MLP cpp
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-20 13:42:10
Last modified: 2016-07-20 14:56:03
GCC version: 4.7.3
std = C++ 11
******************************************/

#include "../include/MLP.h"
#include "../include/HiddenLayer.h"

MLP::MLP() {}

MLP::MLP(int n_in, int n_out, int n_hidden, vector<vector<double>> input) {
    this->n_in = n_in;
    this->n_out = n_hidden;
    this->n_hidden = n_hidden;
    this->input = input;
    this->hiddenLayer = HiddenLayer(n_in, n_out, input, 0);
    this->logisticLayer = LogisticRegression();
    //this->logisticLayer = LogisticRegression(this->hiddenLayer.output, n_hidden, n_out);
}

MLP::~MLP() {}

int main() {
    return 0;
}
