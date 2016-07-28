//coding:utf-8
/***********************************************************
Program: RBM cpp
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-27 11:03:50
Last modified: 2016-07-28 14:46:50
GCC version: 4.9.3
***********************************************************/

#include <vector>
#include <cmath>

#include "../include/RBM.h"
#include "../include/utils.h"
#include <iostream>

using namespace std;

RBM::~RBM() {}

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

void RBM::sampleHGivenV(vector<vector<double>>& h1_sample, vector<vector<double>>& pre_sigmoid_h1, vector<vector<double>>& h1_mean, vector<vector<double>>& v0_sample) {
    //the parameters order:
    //parameter: output, pre, mean, input

    dot(v0_sample, weights, pre_sigmoid_h1, hbias);
    if(h1_mean.empty())
        h1_mean = vector<vector<double>>(pre_sigmoid_h1.size(), vector<double>(pre_sigmoid_h1[0].size(), 0));
    if(h1_sample.empty())
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

void RBM::sampleVGivenH(vector<vector<double>>& v1_sample, vector<vector<double>>& pre_sigmoid_v1, vector<vector<double>>& v1_mean, vector<vector<double>>& h0_sample) {
    //sample_results, pre_si, mean_of_2nd_parameter, input
    //parameter: output, pre, mean, input
    vector<vector<double>> T;
    transpose(weights, T);
    dot(h0_sample, T, pre_sigmoid_v1, vbias);

    if(v1_mean.empty())
        v1_mean = vector<vector<double>>(pre_sigmoid_v1.size(), vector<double>(pre_sigmoid_v1[0].size(), 0));
    if(v1_sample.empty())
        v1_sample = vector<vector<double>>(pre_sigmoid_v1.size(), vector<double>(pre_sigmoid_v1[0].size(), 0));

    for(unsigned int i = 0; i < pre_sigmoid_v1.size(); ++i)
        for(unsigned int j = 0; j < pre_sigmoid_v1[0].size(); ++j)
            v1_mean[i][j] = sigmoid(pre_sigmoid_v1[i][j]);

    //Sample from binomial distribution
    for(unsigned int i = 0; i < v1_mean.size(); ++i)
        for(unsigned int j = 0; j < v1_mean[0].size(); ++j) {
            if(randRange(0, 1) <= v1_mean[i][j])
                v1_sample[i][j] = 1;
            else
                v1_sample[i][j] = 0;
        }
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

void RBM::update(double rate, vector<vector<double>> persistence = vector<vector<double>>{}, int k = 1) {
    //contrasive divergence
    //compute positive phase
    vector<vector<double>> chain_start;

    vector<vector<double>> pre_sigmoid_ph(vector<vector<double>>(input.size(), vector<double>(n_hidden, 0)));
    vector<vector<double>> ph_means(vector<vector<double>>(input.size(), vector<double>(n_hidden, 0)));
    vector<vector<double>> ph_samples(vector<vector<double>>(input.size(), vector<double>(n_hidden, 0)));

    sampleHGivenV(ph_samples, pre_sigmoid_ph, ph_means, input);
    //=======================//
    //

    if(persistence.empty())
        chain_start = ph_samples;
    else
        chain_start = persistence;

    vector<vector<double>> pre_sigmoid_nvs(vector<vector<double>>(input.size(), vector<double>(n_visible, 0)));
    vector<vector<double>> nv_means(vector<vector<double>>(input.size(), vector<double>(n_visible, 0)));
    vector<vector<double>> nv_samples(vector<vector<double>>(input.size(), vector<double>(n_visible, 0)));

    vector<vector<double>> pre_sigmoid_nhs(vector<vector<double>>(input.size(), vector<double>(n_hidden, 0)));
    vector<vector<double>> nh_means(vector<vector<double>>(input.size(), vector<double>(n_hidden, 0)));
    vector<vector<double>> nh_samples(vector<vector<double>>(input.size(), vector<double>(n_hidden, 0)));

    for(int i = 0; i < k; ++i) {
        if(i == 0) {
            sampleVGivenH(nv_samples, pre_sigmoid_nvs, nv_means, chain_start);
            sampleHGivenV(nh_samples, pre_sigmoid_nhs, nh_means, nv_samples);
        } else {
            sampleVGivenH(nv_samples, pre_sigmoid_nvs, nv_means, nh_samples);
            sampleHGivenV(nh_samples, pre_sigmoid_nhs, nh_means, nv_samples);
        }
    }

    for(unsigned int k = 0; k < input.size(); ++k) {
        for(int i = 0; i < n_hidden; ++i) {
            for(int j = 0; j < n_visible; ++j) {
                weights[i][j] += rate * (ph_samples[k][i] * input[k][j] - nh_means[k][i] * nv_samples[k][j]) / input.size();
            }
            hbias[i] += rate * (ph_samples[k][i] - nh_means[k][i]) / input.size();
        }

        for(int i = 0; i < n_visible; ++i)
            vbias[i] += rate * (input[k][i] - nv_samples[k][i]) / input.size();
    }
}

void RBM::reconstruct(vector<vector<double>>& v, vector<vector<double>>& reconstructed_v) {
    if(reconstructed_v.empty())
        reconstructed_v = vector<vector<double>>(v.size(), vector<double>(v[0].size(), 0));
    vector<vector<double>> h;

    dot(v, weights, h, hbias);

    vector<vector<double>> T;
    transpose(weights, T);
    dot(h, T, reconstructed_v, vbias);
}


void test_rbm() {
  double learning_rate = 0.1;
  int training_epochs = 1000;
  
  int n_visible = 6;
  int n_hidden = 3;

  // training data
  vector<vector<double>> train_X = {
    {1, 1, 1, 0, 0, 0},
    {1, 0, 1, 0, 0, 0},
    {1, 1, 1, 0, 0, 0},
    {0, 0, 1, 1, 1, 0},
    {0, 0, 1, 0, 1, 0},
    {0, 0, 1, 1, 1, 0}
  };


  // construct RBM
  RBM rbm(n_visible, n_hidden, train_X, vector<double>(), vector<double>());

  // train
    for(int epoch=0; epoch<training_epochs; epoch++) {
        rbm.update(learning_rate);
    }

  // test data
  vector<vector<double>> test_X = {
    {1, 1, 0, 0, 0, 0},
    {0, 0, 0, 1, 1, 0}
  };


  vector<vector<double>> rec;
  // test
    rbm.reconstruct(test_X, rec);
  for(auto vec: rec) {
    for(auto d: vec)
        cout << d << " ";
    cout << endl;
  }
}

int main() {
    test_rbm();
    return 0;
}
