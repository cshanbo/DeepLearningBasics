//coding:utf-8
/***********************************************************
Program: Restricted Boltzmann Machine header file
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-27 08:54:13
Last modified: 2016-07-27 08:54:13
GCC version: 4.9.3
***********************************************************/

#ifndef _RBM_H_
#define _RBM_H_

#include <vector>
using namespace std;

class RBM {
public:
    vector<vector<double>> weights;
    vector<double> hbias;
    vector<double> vbias;
    int n_visible;
    int n_hidden;
    vector<vector<double>> input;

    RBM();
    ~RBM();
    RBM(int, int, vector<vector<double>>, vector<double>, vector<double>);

    vector<double> freeEnergy(vector<vector<double>>);

    double getCostUpdates(double = 0.1, int = 1);

    void sampleHGivenV(vector<vector<double>> v0_sample, vector<vector<double>>& pre_sigmoid_h1, vector<vector<double>>& h1_mean, vector<vector<double>>& h1_sample);

};

#endif
