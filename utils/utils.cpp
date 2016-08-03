//coding:utf-8
/******************************************
Program: utils cpp
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-20 13:13:40
Last modified: 2016-08-03 19:49:45
GCC version: 4.7.3
std = C++ 11
******************************************/

#include "../include/utils.h"
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <utility>
double randRange(double fMin,  double fMax) {
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
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

void dot(vector<vector<double>>& input, vector<vector<double>>& matrix, vector<vector<double>>& ret, pair<int, int> p1, pair<int, int> p2, vector<double> bias) {
    assert(!(input.empty() || matrix.empty()));
    assert(!((unsigned int)(p2.first - p1.first + 1) != matrix.size() || (bias.size() != 0 && matrix[0].size() != bias.size())));

    //p1 is the corrdination of left-top point of a silde in a matrix. p1.first is the x_axis (col),  p1.second is the y_axis (row), p2 is the corrdination of right-bottom point of a slide in a matrix.
    
    ret = vector<vector<double>>(p2.second - p1.second + 1, vector<double>(matrix[0].size(), 0));
    for(int i = p1.second; i <= p2.second; ++i)
        for(unsigned int j = 0; j < matrix[0].size(); ++j)
            for(int k = p1.first; k <= p2.first; ++k)
                ret[i - p1.second][j] += input[i][k] * matrix[k - p1.first][j];

    if(bias.size() != 0)
        for(unsigned int i = 0; i < ret.size(); ++i)
            for(unsigned int j = 0; j < ret[0].size(); ++j)
                ret[i][j] += bias[j];
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

double cosine(vector<double>& v1, vector<double>& v2) {
    assert(v1.size() == v2.size());
    double ret = 0, v1_l = 0, v2_l = 0;
    for(unsigned int i = 0; i < v1.size(); ++i) {
        ret += v1[i] * v2[i];
        v1_l += v1[i] * v1[i];
        v2_l += v2[i] * v2[i];
    }
    return ret / sqrt(v1_l) / sqrt(v2_l);
}

double maxPooling(matrix<double>& m, pair<int, int> p1, pair<int, int> p2) {
    assert(!m.empty());
    double r = m[p1.second][p1.first];
    for(int i = 0; i <= p2.second; ++i)
        for(int j = 0; j <= p2.first; ++j)
            if(r < m[i][j])
                r = m[i][j];
    return r;
}

double dotElement(matrix<double>& m1, matrix<double>& m2) {
    assert(!m1.empty() && !m2.empty() && m1.size() == m2.size() && m1[0].size() == m2[0].size());
    double ret = 0;
    for(unsigned int i = 0; i < m1.size(); ++i)
        for(unsigned int j = 0; j < m1[0].size(); ++j)
            ret += m1[i][j] * m2[i][j];
    return ret;
}

double dotElement(matrix<double>& m1, matrix<double>& m2, pair<int, int> p1, pair<int, int> p2) {
    //the definition of p1 p2 is exactly same as the dot function
    assert(!m1.empty() && !m2.empty() && p2.second - p1.second + 1 == m2.size() && p2.first - p2.first + 1 == m2[0].size());
    double ret = 0;
    for(unsigned int i = 0; i < m2.size(); ++i)
        for(unsigned int j = 0; j < m2[0].size(); ++j)
            if(p1.second + i < m1.size() && p1.first + j < m1[0].size())
                ret += m1[p1.second + i][p1.first + j] * m2[i][j];
    return ret;
    //if ignoring the border
}
