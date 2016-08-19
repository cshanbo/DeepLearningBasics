//coding:utf-8
/*****************************************
Program: Recurrent NN
Description: 
Author: cshanbo@gmail.com
Date: 2016-08-04 10:53:00
Last modified: 2016-08-17 15:43:48
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

#include <cmath>
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

RNN::RNN(int nh, int nc, int ne, int de, int cs, int bs) {
    //ne is the vocab_size
    //embeddings contains all of the word embedding for vocab
    //
    assert(cs >= 1 && cs % 2 == 1);
    this->nh = nh;
    this->nc = nc;
    this->ne = ne;
    this->de = de;
    this->cs = cs;
    this->back_size = bs;

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
                    for(int m = 0; m < dim_embed; m++)
                        idx++;
                else
                    for(int m = 0; m < dim_embed; m++)
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

void RNN::minibatch(matrix<int>& window_matrix, tensor3<int>& ret) {
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

void RNN::getSentenceLabels(tensor3<double>& s) {
    if(s.empty())
        return;
    for(auto mt: s)
        y_given_x_sentence.push_back(mt[0]);

    y_given_x_lastword = s[s.size() - 1][0];

    for(auto vec: y_given_x_sentence)
        y_pred.push_back(maxIndex(vec));
}

double RNN::sentenceNLL(matrix<double>& y_given_x_sentence, vector<int>& y) {
// calculate the negative loglikelihood of a sentence prediction
    assert(y_given_x_sentence.size() == y.size());
    if(y_given_x_sentence.empty())
        return -1;
    double sum = 0;
    for(unsigned int i = 0; i < y_given_x_sentence.size(); ++i)
        sum += log(y_given_x_sentence[i][y[i]]);
    return -1 * sum / y.size();
}

void RNN::update(matrix<double>& y_given_x_sentence, vector<int>& y, tensor3<double>& h, double rate) {

}

//this embs is the embeddings of whole vocab, not just minibatch
//index is the mini-batch index
void RNN::update(matrix<double>& y_given_x_sentence, vector<int>& y, matrix<double>& embs, vector<int>& index, tensor3<double>& s, tensor3<double>& h, double rate) {
    //mini batch is a sentence, mini batch update
    if(y_given_x_sentence.empty())
       return; 
    //update weights shape (n_hidden, n_output)
    for(unsigned int k = 0; k < y_given_x_sentence.size(); ++k) {
        for(unsigned int i = 0; i < weights.size(); ++i) {
            for(unsigned int j = 0; j < weights[0].size(); ++j) {
                weights[i][j] += rate * (y[k] == 1? y_given_x_sentence[k][j] - 1: y_given_x_sentence[k][j]) * h[k][0][i] / y_given_x_sentence.size();
            }
        }
    }

    //update wx
    for(unsigned int k = 0; k < y_given_x_sentence.size(); ++k) {
        for(unsigned int i = 0; i < wx.size(); ++i) {
            for(unsigned int j = 0; j < wx[0].size(); ++j) {
                wx[i][j] += rate * (y[k] == 1? y_given_x_sentence[k][j] - 1: y_given_x_sentence[k][j]) *  input[k][i] / y_given_x_sentence.size();
            }
        }
    }
    
    //update wh
    for(unsigned int k = 0; k < y_given_x_sentence.size(); ++k) {
        for(unsigned int i = 0; i < wh.size(); ++i) {
            for(unsigned int j = 0; j < wh[0].size(); ++j) {
                if(k > 0)
                    wh[i][j] += rate * (y[k] == 1? y_given_x_sentence[k][j] - 1: y_given_x_sentence[k][j]) * h[k - 1][0][i] / y_given_x_sentence.size();
            }
        }
    }

    //update word embedding
    for(unsigned int k = 0; k < index.size(); ++k) {
        //input is the expanded vector
        for(unsigned int i = 0; i < this->wh.size(); ++i)
            for(unsigned int j = 0; j < input[0].size(); ++j) {
                int idx = j / this->cs; //the idx-th word in a window size
                int idx1 = j % this->cs;
                //the idx-th word's idx1-th element
                int real_idx = k - this->cs / 2 + idx; 
                if(real_idx >= 0 && real_idx < (int)index.size())
                    embs[index[real_idx]][idx1] += rate * (y[k] == 1? y_given_x_sentence[k][j] - 1: y_given_x_sentence[k][j]) * wx[j][i] / index.size();
            }
    }
    //recurrence(embs, h, s);
    normalizeEmbedding(embs);
}

void RNN::normalizeEmbedding(matrix<double>& embs) {
    if(embs.empty())
        return;
    for(auto& vec: embs) {
        double sum;
        for(auto d: vec)
            sum += d * d;
        sum = sqrt(sum);
        for(auto& d: vec)
            d /= sum;
    }
}

int main() {
    vector<int> indexes{4, 1, 8, 3, 5, 2, 6};
    matrix<int> out;
    matrix<double> embeddings{
        {0, 0, 0, 0, 0, 0, 0, 0, 0}, 
        {1, 1, 1, 1, 1, 1, 1, 1, 1}, 
        {2, 2, 2, 2, 2, 2, 2, 2, 2}, 
        {3, 3.1, 3, 3, 3, 3, 3, 3, 3}, 
        {4, 4, 4.2, 4, 4, 4, 4, 4, 4}, 
        {5, 5, 5.1, 5, 5, 5, 5, 5, 5}, 
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
    bs :: bptt size
*/
    RNN rnn(3, 4, 10, 9, 3, 4);

    //rnn.embeddings = embeddings;
    rnn.getWindowMatrix(indexes, out);
    
    tensor3<int> ret;
    //get mini batch for bptt
    rnn.minibatch(out, ret);

    tensor3<double> embs;

    rnn.getEmbeddingsFromIndex(ret, embs);

    rnn.input.clear();
    for(auto matrix: embs)
        rnn.input.push_back(matrix[matrix.size() - 1]);
    print(rnn.input);

    //for(auto m: embs)
    //    print(m);

    tensor3<double> h, s;
    rnn.recurrence(embs, h, s);

    int i = 0;
    for(auto ma: h) {
        cout << i++ << endl;
        print(ma);
    }

    rnn.getSentenceLabels(s);
    //print(rnn.y_given_x_sentence);
    //for(auto d: rnn.y_given_x_lastword)
    //    cout << d << " ";
    //cout << endl;
    //for(auto i: rnn.y_pred)
    //    cout << i << " ";
    //cout << endl;

    return 0;
}











