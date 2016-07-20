//coding:utf-8
/*
Program: Hidden Layer header
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-20 08:33:07
Last modified: 2016-07-20 08:33:07
GCC version: 4.7.3
*/

#ifndef _HIDDEN_LAYER_H_
#define _HIDDEN_LAYER_H_

#include <iostream>
#include <vector>
#include <string>
using namespace std;

class HiddenLayer {
    int n_in;       //dimension of input
    int n_out;	    //dimension of output
    vector<vector<double>> weights;     //a double matrix
    vector<double> bias;                //a double vector
    vector<vector<double>> input;       //a double matrix
public:
    void activation(vector<double>&, string);
    HiddenLayer(int, int, vector<vector<double>>);
    ~HiddenLayer();
};

#endif
