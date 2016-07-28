//coding:utf-8
/******************************************
Program: utils cpp
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-20 13:13:40
Last modified: 2016-07-28 08:36:12
GCC version: 4.7.3
std = C++ 11
******************************************/

#include "../include/utils.h"
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
double randRange(double fMin,  double fMax) {
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(x));
}

int maxIndex(vector<double>& vec) {
    if(vec.empty())
        return -1;
    int ret = 0;
    double m = 0;
    for(unsigned int i = 0; i < vec.size(); ++i)
        if(vec[i] > m) {
            m = vec[i];
            ret = i;
        }
    return ret;
}

void dot(vector<vector<double>>& input, vector<vector<double>>& matrix, vector<vector<double>>& ret, vector<double> bias) {
    assert(!(input.empty() || matrix.empty()));
    assert(!(input[0].size() != matrix.size() || (bias.size() != 0 && matrix[0].size() != bias.size())));

    ret = vector<vector<double>>(input.size(), vector<double>(matrix[0].size(), 0));
    for(unsigned int i = 0; i < input.size(); ++i)
        for(unsigned int j = 0; j < matrix[0].size(); ++j)
            for(unsigned int k = 0; k < input[0].size(); ++k)
                ret[i][j] += input[i][k] * matrix[k][j];

    if(bias.size() != 0)
        for(unsigned int i = 0; i < ret.size(); ++i)
            for(unsigned int j = 0; j < ret[0].size(); j++)
                ret[i][j] += bias[j];
    return;
}

void print(vector<vector<double>> vec) {
    if(vec.empty())
        cout << "empty" << endl;
    for(auto v: vec) {
        for(auto d: v)
          cout << d << " "; 
        cout << endl;
    }
}
double L1(vector<vector<double>>& w1) {
    double ret = 0;
    for(auto v: w1)
        for(auto d: v)
            ret += abs(d);
    return ret;
}

double L2(vector<vector<double>>& w1) {
    double ret = 0;
    for(auto v: w1)
        for(auto d: v)
            ret += d * d;
    return sqrt(ret);
}

void split(const string &src, const string &separator, vector<string>& dest_list) {
    size_t pre_pos = 0, position;
    string temp = "";
    dest_list.clear();
    if(src.empty())
        return;
    while((position = src.find(separator.c_str(), pre_pos)) != string::npos) {
        temp.assign(src, pre_pos, position - pre_pos);
        if(!temp.empty())
            dest_list.push_back(temp);
        pre_pos = position + separator.length();
    }
    temp.assign(src, pre_pos, src.length() - pre_pos);
    if(!temp.empty())
        dest_list.push_back(temp);
}

string &trim(string &line) {
    if(!line.empty()) {
        string empty_str = "\t\r\n ";
        line.erase(0, line.find_first_not_of(empty_str));
        line.erase(line.find_last_not_of(empty_str) + 1);
    }
    return line;
}

void string_replace(string &origin, const string &src, const string &tgt) {
    string::size_type pos = 0, srclen = src.size(),  dstlen = tgt.size();
    while((pos = origin.find(src, pos)) != string::npos) {
        origin.replace(pos, srclen, tgt);
        pos += dstlen;
    }
}

void transpose(vector<vector<double>> src, vector<vector<double>> &tgt) {
    if(src.empty())
        return;
    tgt = vector<vector<double>>(src[0].size(), vector<double>(src.size(), 0));
    for(unsigned int i = 0; i < src.size(); ++i)
        for(unsigned int j = 0; j < src[0].size(); ++j)
            tgt[j][i] = src[i][j];
}
