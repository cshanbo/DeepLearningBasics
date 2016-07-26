//coding:utf-8
/******************************************
Program: MLP cpp
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-20 13:42:10
Last modified: 2016-07-25 21:00:07
GCC version: 4.7.3
std = C++ 11
******************************************/

#include <cmath>
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
    this->hiddenLayer = HiddenLayer(n_in, n_hidden, input, 0);
    this->logisticLayer = LogisticRegression(this->hiddenLayer.output, n_hidden, n_out);
}

double MLP::l1_norm(vector<vector<double>>& w1, vector<vector<double>>& w2) {
    return L1(w1) + L1(w2);
}

double MLP::l2_norm(vector<vector<double>>& w1, vector<vector<double>>& w2) {
    return L2(w1) + L2(w2);
}

double MLP::cost(vector<int> y, double l1_rate, double l2_rate) {
    double ret = logisticLayer.negativeLogLikelihood(y) + l1_rate * l1_norm(hiddenLayer.weights, logisticLayer.weights) + l2_rate * l2_norm(hiddenLayer.weights, logisticLayer.weights);
    return ret;
}

void MLP::update(double rate, double l1_rate, double l2_rate, vector<int> y) {
    /*cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    gparams = [T.grad(cost, param) for param in classifier.params]
    */

    //update logistic layer first (output layer)
    //other than the original logistic layer, there is another updating at the l1 and l2 normalization
    //the gradient of cost is the sum of negative_log_likelihood, l1_norm, l2_norm
    
    //1. update by negative_log_likelihood, which is exact same as logistic layer update
    //in this update, there is hidden layer involved
    //so I need to modify the gradient calculation
    //
    double loss = cost(y, l1_rate, l2_rate);
    logisticLayer.update(rate, y);
    //2. update L1 and l2 norm. Both logistic and hidden layer should be updated
    double sumLogistic = 0, sumHidden = 0;

    for(auto vec: logisticLayer.weights)
        for(auto d: vec)
            sumLogistic += d * d;

    for(auto vec: hiddenLayer.weights)
        for(auto d: vec)
            sumHidden += d * d;

    for(unsigned int i = 0; i < logisticLayer.weights.size(); ++i) {
        for(unsigned int j = 0; j < logisticLayer.weights[0].size(); ++j) {
            logisticLayer.weights[i][j] -= rate * loss * l1_rate * (logisticLayer.weights[i][j] < 0? -1 : 1); 
            logisticLayer.weights[i][j] -= rate * loss * l2_rate * logisticLayer.weights[i][j] / sqrt(sumLogistic); 
        }
    }


    for(unsigned int i = 0; i < hiddenLayer.weights.size(); ++i) {
        for(unsigned int j = 0; j < hiddenLayer.weights[0].size(); ++j) {
            hiddenLayer.weights[i][j] -= rate * l1_rate * loss * (hiddenLayer.weights[i][j] < 0? -1 : 1); 
            hiddenLayer.weights[i][j] -= rate * l2_rate * loss * hiddenLayer.weights[i][j] / sqrt(sumHidden); 
        }
    }

    dot(input, hiddenLayer.weights, hiddenLayer.output, hiddenLayer.bias);
    hiddenLayer.activation(hiddenLayer.output, 0);
    dot(hiddenLayer.output, logisticLayer.weights, logisticLayer.y_given_x, logisticLayer.bias);
    logisticLayer.softmax(logisticLayer.y_given_x);
    //logisticLayer.softmax(logisticLayer.output);
}

int main() {
    vector<vector<double>> input{
        {1, 1, 1, 0, 0, 0},
        {1, 0, 1, 0, 0, 0},
        {1, 1, 1, 0, 0, 0},
        {0, 0, 1, 1, 1, 0},
        {0, 0, 1, 1, 0, 0},
        {0, 0, 1, 1, 1, 0}, 
        {0, 0, 0, 0, 0, 1}, 
    };

    vector<vector<double>> test{
        {1, 0, 1, 0, 0, 0},
        {0, 0, 1, 1, 1, 0}, 
        {0, 0, 0, 1, 1, 1}, 
        {0, 1, 0, 0, 0, 0}, 
        {0, 1, 1, 1, 1, 1}, 
        {0, 0, 0, 1, 0, 1}, 
    };

    vector<int> ytrain{0, 0, 0, 1, 1, 1, 1};

    vector<int> ytest{0, 1, 1, 0, 1, 1};

    MLP feedforward(6, 2, 4, input); //n_in, n_out, n_hidden
    for(int i = 0; i < 500; ++i)
        feedforward.update(0.1, 0, 0, ytrain);

    print(feedforward.hiddenLayer.weights);
    vector<int> label;
    vector<vector<double>> ygx;
    dot(test, feedforward.hiddenLayer.weights, ygx, feedforward.hiddenLayer.bias);
    feedforward.hiddenLayer.activation(ygx, 0);
    feedforward.logisticLayer.test(feedforward.hiddenLayer.output, ytest);
    
    return 0;
}
