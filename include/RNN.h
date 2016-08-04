//coding:utf-8
/*****************************************
Program: Recurrent neural network
Description: 
Author: cshanbo@gmail.com
Date: 2016-08-04 10:36:07
Last modified: 2016-08-04 10:36:07
GCC version: 4.9.3
*****************************************/

#ifndef _RNN_H_
#define _RNN_H_

#include "../include/utils.h"
using namespace std;

class RNN {
public:
    matrix<double> embeddings;
    matrix<double> wx;
    matrix<double> wh;
    matrix<double> weights;

    vector<double> hbias;
    vector<double> bias;
    vector<double> h0;

    RNN();
    ~RNN();
    /*
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size
    */
    RNN(int nh, int nc, int ne, int de, int cs);

    void getEmbeddingsFromIndex(matrix<int>&, matrix<double>&);

    void getWindowMatrix(vector<int>&, matrix<int>&, int = 7);
};

#endif
