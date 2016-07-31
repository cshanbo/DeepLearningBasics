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

#include <iostrea>
using namespace std;

class Autoencoder {
public:
    int n_visible;
    int n_hidden;
    vector<vector<double>> weights;
    vector<double> vbias;
    vector<double> hbias;

    Autoencoder(int, int, );
    ~Autoencoder();
};

#endif
