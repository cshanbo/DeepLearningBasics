//coding:utf-8
/***********************************************************
Program: RBM cpp
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-27 11:03:50
Last modified: 2016-07-27 15:59:10
GCC version: 4.9.3
***********************************************************/

#include <vector>
#include <cmath>

#include "../include/RBM.h"
#include "../include/utils.h"

using namespace std;

RBM::RBM(int n_visible, int n_hidden, vector<vector<double>> input, vector<double> vbias, vector<double> hbias) {
    this->n_hidden = n_hidden;
    this->n_visible = n_visible;
    if(vbias.empty())
        this->vbias = vector<double>(n_visible, 0);
    else
        this->vbias = vbias;
    if(hbias.empty())
        this->hbias = vector<double>(n_hidden, 0);
    else
        this->hbias = hbias;
    this->input = input;
    this->weights = vector<vector<double>> (n_visible, vector<double>(n_hidden, 0));
    for(unsigned int i = 0; i < this->weights.size(); ++i) 
        for(unsigned int j = 0; j < this->weights[0].size(); ++j)
            this->weights[i][j] = randRange(-4.0 * sqrt(6.0 / (n_visible + n_hidden)), 4.0 * sqrt(6.0 / (n_visible + n_hidden)));
}

vector<double> RBM::freeEnergy(vector<vector<double>> v_sample) {
    vector<vector<double>> wx_b;
    dot(v_sample, weights, wx_b, hbias);
    
    vector<vector<double>> vbias_term;
    vector<vector<double>> transposed;
    transpose(vector<vector<double>>{vbias}, transposed);
    dot(v_sample, transposed, vbias_term);

    vector<double> hidden_term; 
    for(unsigned int i = 0; i < wx_b.size(); ++i) {
        double sum = 0;
        for(unsigned int j = 0; j < wx_b[0].size(); ++j)
            sum += log(1 + exp(wx_b[i][j]));
        hidden_term.push_back(sum);
    }

    vector<double> ret;
    for(unsigned int i = 0; i < hidden_term.size(); i++)
        ret.push_back(hidden_term[i] - vbias_term[i][0]);
    return ret;
}

void RBM::sampleHGivenV(vector<vector<double>> v0_sample, vector<vector<double>>& pre_sigmoid_h1, vector<vector<double>>& h1_mean, vector<vector<double>>& h1_sample) {
    dot(v0_sample, weights, pre_sigmoid_h1, hbias);
    h1_mean = vector<vector<double>>(pre_sigmoid_h1.size(), vector<double>(pre_sigmoid_h1[0].size(), 0));
    h1_sample = vector<vector<double>>(pre_sigmoid_h1.size(), vector<double>(pre_sigmoid_h1[0].size(), 0));

    for(unsigned int i = 0; i < pre_sigmoid_h1.size(); ++i)
        for(unsigned int j = 0; j < pre_sigmoid_h1[0].size(); ++j)
            h1_mean[i][j] = sigmoid(pre_sigmoid_h1[i][j]);

    //Sample from binomial distribution
    for(unsigned int i = 0; i < h1_mean.size(); ++i)
        for(unsigned int j = 0; j < h1_mean[0].size(); ++j) {
            if(randRange(0, 1) <= h1_mean[i][j])
                h1_sample[i][j] = 1;
            else
                h1_sample[i][j] = 0;
        }
}

int main() {
    return 0;
}
