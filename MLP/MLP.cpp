//coding:utf-8
/******************************************
Program: MLP cpp
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-20 13:42:10
Last modified: 2016-07-26 19:51:01
GCC version: 4.7.3
std = C++ 11
******************************************/

#include <cmath>
#include <cassert>
#include "../include/MLP.h"
#include "../include/HiddenLayer.h"
#include "../include/utils.h"

MLP::MLP() {}

MLP::~MLP() {}

MLP::MLP(int n_in, int n_out, int n_hidden, vector<vector<double>> input) {
    this->n_in = n_in;
    this->n_out = n_out;
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
    assert(!y.empty());
    double dy = 0;
    for(unsigned int k = 0; k < input.size(); ++k) {
        for(int i = 0; i < n_out; ++i) {
            dy = y[k] == i? (logisticLayer.y_given_x[k][i] - 1): (logisticLayer.y_given_x[k][i]);
            for(int j = 0; j < n_hidden; ++j) {
                logisticLayer.weights[j][i] -= rate * dy * hiddenLayer.output[k][j] / input.size();
            }
            logisticLayer.bias[i] -= rate * dy / input.size();
        }
    }
    //update the hidden layer

    dy = 0;
    for(unsigned int k = 0; k < input.size(); ++k)
        for(int i = 0; i < n_out; ++i) {
            dy = y[k] == i? logisticLayer.y_given_x[k][i] - 1: logisticLayer.y_given_x[k][i];
            for(int m = 0; m < n_hidden; ++m) {
                for(int j = 0; j < n_in; ++j)
                    hiddenLayer.weights[j][m] -= rate * dy * (1 - pow(tanh(hiddenLayer.output[k][i]), 2)) * input[k][j] * logisticLayer.weights[m][i] / input.size();
                hiddenLayer.bias[m] -= rate * dy * (1 - pow(tanh(hiddenLayer.output[k][i]), 2)) * logisticLayer.weights[m][i] / input.size();
            }
        }
    //2. update L1 and l2 norm. Both logistic and hidden layer should be updated
    //there might be something wrong with this part
    //not entirely sure what the problem is
    //use l1_rate and l2_rate as 0 to avoid the influence
    //use small hyper parameters to avoid over-fitting

    for(unsigned int i = 0; i < logisticLayer.weights.size(); ++i) {
        for(unsigned int j = 0; j < logisticLayer.weights[0].size(); ++j) {
            logisticLayer.weights[i][j] -= rate * l1_rate * (logisticLayer.weights[i][j] < 0? -1 : 1); 
            logisticLayer.weights[i][j] -= rate * l2_rate * 2 * logisticLayer.weights[i][j]; 
        }
    }


    for(unsigned int i = 0; i < hiddenLayer.weights.size(); ++i) {
        for(unsigned int j = 0; j < hiddenLayer.weights[0].size(); ++j) {
            hiddenLayer.weights[i][j] -= rate * l1_rate * (hiddenLayer.weights[i][j] < 0? -1 : 1); 
            hiddenLayer.weights[i][j] -= rate * l2_rate * 2 * hiddenLayer.weights[i][j]; 
        }
    }

    //forward
    dot(input, hiddenLayer.weights, hiddenLayer.output, hiddenLayer.bias);

    hiddenLayer.activation(hiddenLayer.output, 0);

    dot(hiddenLayer.output, logisticLayer.weights, logisticLayer.y_given_x, logisticLayer.bias);

    logisticLayer.softmax(logisticLayer.y_given_x);

    for(unsigned int i = 0; i < logisticLayer.y_pred.size(); ++i)
        logisticLayer.y_pred[i] = maxIndex(logisticLayer.y_given_x[i]);
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
    for(int i = 0; i < 800; ++i) {
        feedforward.update(0.1, 0.01, 0.01, ytrain);
        cout << feedforward.logisticLayer.negativeLogLikelihood(ytrain) << endl;
    }

    vector<int> label;
    vector<vector<double>> ygx;

    dot(test, feedforward.hiddenLayer.weights, ygx, feedforward.hiddenLayer.bias);

    feedforward.hiddenLayer.activation(ygx, 0);

    feedforward.logisticLayer.test(ygx, ytest);
    return 0;
}
