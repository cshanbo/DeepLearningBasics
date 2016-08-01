//coding:utf-8
/***********************************************************
Program: Auto Encoder header
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-27 10:26:29
Last modified: 2016-07-27 10:26:29
GCC version: 4.9.3
***********************************************************/

#ifndef _AUTO_ENCODER_H_
#define _AUTO_ENCODER_H_

#include <iostream>
#include <vector>
using namespace std;

class Autoencoder {
public:
    int n_visible;
    int n_hidden;
    vector<vector<double>> weights;
    vector<vector<double>> input;
    vector<double> vbias;
    vector<double> hbias;

    Autoencoder(vector<vector<double>>, int, int);
    Autoencoder();
    ~Autoencoder();

    void getHiddenValues(vector<vector<double>>&, vector<vector<double>>&);

    void getReconstructedInput(vector<vector<double>>&, vector<vector<double>>&);

    void getCorruptedInput(vector<vector<double>>, vector<vector<double>>&, double);

    void update(double, double);
};

#endif
