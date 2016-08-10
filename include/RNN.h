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
    matrix<double> embeddings;  //the collection of all word embeddings, shape is the vocab_size + 1 * embedding_dim
    matrix<double> wx;  //shape is embedding_dim * window_sz 
    matrix<double> wh;
    matrix<double> weights;

    //one mini batch
    matrix<double> input;

    vector<double> hbias;
    vector<double> bias;
    vector<double> h0;

    //y_given_x_sentence (a mini batch)
    matrix<double> y_given_x_sentence;
    //the last output 
    vector<double> y_given_x_lastword;
    //the prediction of a mini batch
    vector<int> y_pred;

    int nh;
    int nc;
    int ne;
    int de;
    int cs;
    int back_size;

    RNN();

    ~RNN();

    /*
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size
        bs :: bptt size
    */

    RNN(int, int, int, int, int, int);

    void getEmbeddingsFromIndex(matrix<int>&, matrix<double>&);

    void getEmbeddingsFromIndex(tensor3<int>& indexes, tensor3<double>& embs);

    void getWindowMatrix(vector<int>&, matrix<int>&, int = 7);

    void minibatch(matrix<int>&, tensor3<int>&);

    void recurrence(tensor3<double>&, tensor3<double>&, tensor3<double>&);

    //s_output, y_given_x_sentence, y_given_x_lastword, y_pred
    void getSentenceLabels(tensor3<double>&);
    // y_given_x_sentence,  y
    double sentenceNLL(matrix<double>&, vector<int>&);

    void update(matrix<double>&, vector<int>&, tensor3<double>&, double);

    void update(matrix<double>&, vector<int>&, matrix<double>&, tensor3<double>&, tensor3<double>&, double);

    void normalizeEmbedding(matrix<double>&);

};

#endif
