//coding:utf-8
/*****************************************
Program: Recurrent NN
Description: 
Author: cshanbo@gmail.com
Date: 2016-08-04 10:53:00
Last modified: 2016-08-04 14:59:51
GCC version: 4.9.3
*****************************************/

/*
 * The followin (Elman) recurrent neural network (E-RNN) takes as input the current input (time t) and the previous hiddent state (time t-1). Then it iterates.
 *
 * In the previous section,  we processed the input to fit this sequential/temporal structure.
 * It consists in a matrix where the row 0 corresponds to the time step t=0,  the row 1 corresponds to the time step t=1,  etc.
 *
 * The parameters of the E-RNN to be learned are:
 *      the word embeddings (real-valued matrix)
 *      the initial hidden state (real-value vector)
 *      two matrices for the linear projection of the input t and the previous hidden layer state t-1
 *      (optional) bias. Recommendation: dont use it.
 *      softmax classification layer on top
 *
 * The hyperparameters define the whole architecture:
 *      dimension of the word embedding
 *      size of the vocabulary
 *      number of hidden units
 *      number of classes
 *      random seed + way to initialize the model
 * */

#include "../include/RNN.h"
#include "../include/utils.h"
#include <vector>
#include <cassert>
#include <iostream>

using namespace std;

/*
    nh :: dimension of the hidden layer
    nc :: number of classes (output labels)
    ne :: number of word embeddings in the vocabulary
    de :: dimension of the word embeddings
    cs :: word window context size
*/
RNN::RNN() {}

RNN::~RNN() {}

RNN::RNN(int nh, int nc, int ne, int de, int cs) {
    //ne is the vocab_size
    //embeddings contains all of the word embedding for vocab
    //
    assert(cs >= 1 && cs % 2 == 1);
    this->embeddings = matrix<double>(ne + 1, vector<double>(de, 0));
    this->wx = matrix<double>(de, vector<double>(cs, 0));
    this->wh = matrix<double>(nh, vector<double>(nh, 0));
    this->weights = matrix<double>(nh, vector<double>(nc, 0));
    this->hbias = vector<double>(nh, 0);
    this->bias = vector<double>(nc, 0);
    this->h0 = vector<double>(nh, 0);

    //initialization
    for(auto& vec: embeddings)
        for(auto& d: vec)
            d = randRange(-0.2, 0.2);


    for(auto& vec: wx)
        for(auto& d: vec)
            d = randRange(-0.2, 0.2);

    for(auto& vec: wh)
        for(auto& d: vec)
            d = randRange(-0.2, 0.2);

    for(auto& vec: weights)
        for(auto& d: vec)
            d = randRange(-0.2, 0.2);
}

void RNN::getEmbeddingsFromIndex(matrix<int>& indexes, matrix<double>& embs) {
    embs = matrix<double>(indexes.size(), vector<double>(this->wx.size() * this->wx[0].size(), 0));
    //indexes:
    //  row number is the number of words in a mini-batch (sentence)
    //  col number is window size
    //the ith word of a sentence, one sentence is a mini batch for this example code 
    for(unsigned int i = 0; i < indexes.size(); ++i) {
        int idx = 0;
        for(unsigned int j = 0; j < indexes[0].size(); ++j) {
            if(indexes[i][j] < 0)
                for(unsigned int k = 0; k < this->wx.size(); k++)
                    idx++;
            else
                for(unsigned int k = 0; k < this->wx.size(); k++)
                    embs[i][idx++] = this->embeddings[indexes[i][j]][k];
        }
    }
}

void RNN::getWindowMatrix(vector<int>& indexes, matrix<int>& out, int w_sz) {
    //index is a vector, which contains the indexes of words in a sentence sequentially
    out = matrix<int>(indexes.size(), vector<int>(w_sz, 0));
    for(unsigned int i = 0; i < indexes.size(); ++i) {
        int idx = 0;
        for(int j = i - w_sz / 2; j <= (int)i + w_sz / 2; ++j) {
            if(j < 0 || j >= (int)indexes.size())
                out[i][idx++] = -1;
            else
                out[i][idx++] = indexes[j];
        } 
    }
}

void recurrence(matrix<double>& x_input, matrix<double>& weight) {
    matrix<double> ret;
    dot(x_input, weight, ret);
}

int main() {
    vector<int> indexes{4, 1, 8, 3, 5};
    matrix<int> out;
    matrix<double> embeddings{
        {0, 0, 0, 0, 0, 0, 0, 0, 0}, 
        {1, 1, 1, 1, 1, 1, 1, 1, 1}, 
        {2, 2, 2, 2, 2, 2, 2, 2, 2}, 
        {3, 3, 3, 3, 3, 3, 3, 3, 3}, 
        {4, 4, 4, 4, 4, 4, 4, 4, 4}, 
        {5, 5, 5, 5, 5, 5, 5, 5, 5}, 
        {6, 6, 6, 6, 6, 6, 6, 6, 6}, 
        {7, 7, 7, 7, 7, 7, 7, 7, 7},
        {8, 8, 8, 8, 8, 8, 8, 8, 8}, 
        {9, 9, 9, 9, 9, 9, 9, 9, 9}
    };

    RNN rnn(3, 3, 10, 9, 7);
    rnn.embeddings = embeddings;
    rnn.getWindowMatrix(indexes, out);

    matrix<double> embs;
    rnn.getEmbeddingsFromIndex(out, embs);

    return 0;
}











