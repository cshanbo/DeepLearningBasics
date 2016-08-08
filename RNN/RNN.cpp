//coding:utf-8
/*****************************************
Program: Recurrent NN
Description: 
Author: cshanbo@gmail.com
Date: 2016-08-04 10:53:00
Last modified: 2016-08-06 09:58:01
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
    this->nh = nh;
    this->nc = nc;
    this->ne = ne;
    this->de = de;
    this->cs = cs;

    this->embeddings = matrix<double>(ne + 1, vector<double>(de, 0));
    this->wx = matrix<double>(de * cs, vector<double>(nh, 0));
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
    embs = matrix<double>(indexes.size(), vector<double>(this->wx.size(), 0));
    //indexes:
    //  row number is the number of words in a mini-batch (sentence)
    //  col number is window size
    //the ith word of a sentence, one sentence is a mini batch for this example code 
    for(unsigned int i = 0; i < indexes.size(); ++i) {
        int idx = 0;
        for(unsigned int j = 0; j < indexes[0].size(); ++j) {
            if(indexes[i][j] < 0)
                for(unsigned int k = 0; k < this->embeddings[0].size(); k++)
                    idx++;
            else
                for(unsigned int k = 0; k < this->embeddings[0].size(); k++)
                    embs[i][idx++] = this->embeddings[indexes[i][j]][k];
        }
    }
}

void RNN::getEmbeddingsFromIndex(tensor3<int>& indexes, tensor3<double>& embs) {
    embs = tensor3<double>();
    int dim_embed = this->embeddings[0].size();
    for(unsigned int i = 0; i < indexes.size(); ++i) {
        matrix<double> temp;
        for(unsigned int j = 0; j < indexes[i].size(); ++j) {
            vector<double> one(indexes[i][j].size() * dim_embed, 0);
            temp.push_back(one);
        }
        embs.push_back(temp);
    }
    //indexes:
    //  row number is the number of words in a mini-batch (sentence)
    //  col number is window size
    //the ith word of a sentence, one sentence is a mini batch for this example code 
    for(unsigned int k = 0; k < indexes.size(); ++k) {
        for(unsigned int i = 0; i < indexes[k].size(); ++i) {
            int idx = 0;
            for(unsigned int j = 0; j < indexes[k][i].size(); ++j) {
                if(indexes[k][i][j] < 0)
                    for(unsigned int m = 0; m < dim_embed; m++)
                        idx++;
                else
                    for(unsigned int m = 0; m < dim_embed; m++)
                        embs[k][i][idx++] = this->embeddings[indexes[k][i][j]][m];
            }
        }
    }
}

void RNN::getWindowMatrix(vector<int>& indexes, matrix<int>& out, int w_sz) {
    //index is a vector, which contains the indexes of words in a sentence sequentially
    w_sz = this->cs;
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

void RNN::minibatch(matrix<int>& window_matrix, tensor3<int>& ret, int back_size) {
    if(window_matrix.empty())
        return;
    ret = tensor3<int>(window_matrix.size(), matrix<int>());
    for(unsigned int i = 0; i < ret.size(); ++i) {
        matrix<int> one;
        for(int j = back_size - 1; j >= 0; --j) {
            if((int)i - j >= 0) 
                one.push_back(window_matrix[(int)i - j]);
        }
        ret[i] = one;
    }
}

void RNN::recurrence(tensor3<double>& x, tensor3<double>& h, tensor3<double>& s) {
    if(x.empty())
        return;
    h = tensor3<double>(x.size(), matrix<double>());
    s = tensor3<double>(x.size(), matrix<double>());
    //from the first to the last word in a batch
    for(unsigned int i = 0; i < x.size(); ++i) {
        if(i == 0) {
            matrix<double> h_1 = matrix<double>(x[i].size(), vector<double>(this->wx[0].size(), 0));
            matrix<double> temp1, temp2;
            dot(x[i], this->wx, temp1);
            dot(h_1, this->wh, temp2, this->hbias);

            h[i] = matrix<double>(temp1.size(), vector<double>(temp1[0].size(), 0));
            for(unsigned int j = 0; j < temp1.size(); ++j)
                for(unsigned int k = 0; k < temp1[0].size(); ++k)
                    h[i][j][k] = temp1[j][k] + temp2[j][k];

            dot(h[i], this->weights, s[i], this->bias);
            for(auto &v: s[i])
                softmax(v);
        } else {
            //calc h_i from x_i and h_i-1
            matrix<double> temp1, temp2;
            matrix<double> first = matrix<double>{x[i][x[i].size() - 1]};
            dot(first, this->wx, temp1);
            dot(h[i - 1], this->wh, temp2, this->hbias);
            h[i] = matrix<double>(temp1.size(), vector<double>(temp1[0].size(), 0));
            for(unsigned int j = 0; j < temp1.size(); ++j)
                for(unsigned int k = 0; k < temp1[0].size(); ++k)
                    h[i][j][k] = temp1[j][k] + temp2[j][k];
            //calc s_i
            dot(h[i], this->weights, s[i], this->bias);

            for(auto &v: s[i])
                softmax(v);
        }
    }
}

void RNN::getSentenceLabels(tensor3<double>& s, matrix<double>& y_given_x_sentence, vector<double>& y_given_x_lastword, vector<int>& y_pred) {
    if(s.empty())
        return;
    for(auto mt: s)
        y_given_x_sentence.push_back(mt[0]);

    y_given_x_lastword = s[s.size() - 1][0];

    for(auto vec: y_given_x_sentence) {
        double max = 0;
        double idx = 0;
        for(unsigned int i = 0; i < vec.size(); ++i)
            if(max < vec[i]) {
                max = vec[i];
                idx = i;
            }
        y_pred.push_back(idx);
    }
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

/*
    nh :: dimension of the hidden layer
    nc :: number of classes (output labels)
    ne :: number of word embeddings in the vocabulary
    de :: dimension of the word embeddings
    cs :: word window context size
*/
    RNN rnn(5, 4, 10, 9, 3);
    rnn.embeddings = embeddings;
    rnn.getWindowMatrix(indexes, out);
    
    print(out);
    tensor3<int> ret;
    //get mini batch for bptt
    rnn.minibatch(out, ret, 4);
    tensor3<double> embs;

    rnn.getEmbeddingsFromIndex(ret, embs);

    for(auto m: embs)
        print(m);
    cout << "---------------" << endl;

    tensor3<double> h, s;
    rnn.recurrence(embs, h, s);
    int i = 0;
    for(auto ma: s) {
        cout << i++ << endl;
        print(ma);
    }

    return 0;
}











