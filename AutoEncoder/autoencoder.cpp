//coding:utf-8
/***********************************************************
Program: AE cpp
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-08-01 09:28:54
Last modified: 2016-08-02 11:14:47
GCC version: 4.9.3
***********************************************************/

#include <iostream>
#include <vector>
#include <cmath>

#include "../include/autoencoder.h"
#include "../include/utils.h"

Autoencoder::Autoencoder() {}

Autoencoder::~Autoencoder() {}

Autoencoder::Autoencoder(vector<vector<double>> input, int n_visible, int n_hidden) {
    this->n_hidden = n_hidden;
    this->n_visible = n_visible;
    this->weights = vector<vector<double>>(n_visible, vector<double>(n_hidden, 0));
    for(auto& vec: this->weights)
        for(auto& d: vec)
            d = randRange(-4.0 * sqrt(6.0 / (n_visible + n_hidden)),  4.0 * sqrt(6.0 / (n_visible + n_hidden)));
    this->hbias = vector<double>(n_hidden, 0);
    this->vbias = vector<double>(n_visible, 0);
    this->input = input;
}

void Autoencoder::getHiddenValues(vector<vector<double>>& input, vector<vector<double>>& hiddenVals) {
    dot(input, weights, hiddenVals, hbias);
    for(auto &vec: hiddenVals)
        for(auto &d: vec)
            d = sigmoid(d);
}

void Autoencoder::getReconstructedInput(vector<vector<double>>& hidden, vector<vector<double>>& rec) {
    vector<vector<double>> trans;
    transpose(weights, trans);
    dot(hidden, trans, rec, vbias);
    for(auto& vec: rec)
        for(auto& d: vec)
            d = sigmoid(d);
}

void Autoencoder::getCorruptedInput(vector<vector<double>> input, vector<vector<double>>& cVec, double corrupt) {
    // sample 
    if(cVec.empty())
        cVec = input;
    for(unsigned int i = 0; i < input.size(); ++i)
        for(unsigned int j = 0; j < input[0].size(); ++j) {
            double rand = randRange(0, 1);
            if(rand > 1 - corrupt)
                cVec[i][j] = 0;
            else
                cVec[i][j] = input[i][j];
        }
}

void Autoencoder::update(double corrupt, double rate) {
    vector<vector<double>> tilde_x;
    getCorruptedInput(input, tilde_x, corrupt);

    vector<vector<double>> y;
    getHiddenValues(tilde_x, y);

    vector<vector<double>> z;
    getReconstructedInput(y, z);

    vector<double> L;
    double allSum = 0;
    for(unsigned int i = 0; i < input.size(); ++i) {
        double sum = 0;
        for(unsigned int j = 0; j < input[0].size(); ++j)
            sum += (input[i][j] * log(z[i][j]) + (1 - input[i][j]) * log(1 - z[i][j]));
        L.push_back(-1 * sum);
        allSum += -1 * sum;
    }

    vector<vector<double>> lv(input.size(), vector<double>(n_visible, 0));
    vector<vector<double>> lh(input.size(), vector<double>(n_hidden, 0));

    //gradient desent
    for(unsigned int k = 0; k < input.size(); ++k) 
        for(unsigned int i = 0; i < vbias.size(); ++i) {
            lv[k][i] = input[k][i] - z[k][i];
            vbias[i] += rate * lv[k][i] / input.size();
        }

    for(unsigned int k = 0; k < input.size(); ++k) {
        for(unsigned int i = 0; i < hbias.size(); ++i) {
            for(unsigned int j = 0; j < vbias.size(); ++j)
                lh[k][i] += weights[j][i] * lv[k][j];
            lh[k][i] *= y[k][i] * (1 - y[k][i]);
            hbias[i] += rate * lh[k][i] / input.size();
        }
    }

    for(unsigned k = 0; k < input.size(); ++k)
        for(unsigned int i = 0; i < weights.size(); ++i)
            for(unsigned int j = 0; j < weights[0].size(); ++j)
                weights[i][j] += rate * (lh[k][j] * tilde_x[k][i] + lv[k][i] * y[k][j]) / input.size();
}

void Autoencoder::reconstruct(vector<vector<double>> &x, vector<vector<double>>& rec) {
    vector<vector<double>> y;
    getHiddenValues(x, y);
    getReconstructedInput(y, rec);
}
