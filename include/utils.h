//coding:utf-8
/*
Program: Utils
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-20 09:30:37
Last modified: 2016-07-20 09:30:37
GCC version: 4.7.3
*/
#ifndef _UTILS_H_
#define _UTILS_H_

#include <string>
#include <vector>
#include <utility>

using namespace std;

template<typename T>
using matrix = vector<vector<T>>;

template<typename T>
using tensor3 = vector<matrix<T>>;

template<typename T>
using tensor4 = matrix<matrix<T>>;

double randRange(double, double);

void dot(matrix<double>&, matrix<double>&, matrix<double>&, vector<double> = vector<double>{});

void dot(matrix<double>&, matrix<double>&, matrix<double>&, pair<int, int> p1, pair<int, int> p2, vector<double> = vector<double>{});

double dotElement(matrix<double>&, matrix<double>&);

double dotElement(matrix<double>&, matrix<double>&, pair<int, int>, pair<int, int>);

void print(matrix<double>);

int maxIndex(vector<double>&);

double sigmoid(double);

double L1(vector<vector<double>>&);

double L2(vector<vector<double>>&);

void split(const string &, const string &, vector<string> &);

string &trim(string &);

void string_replace(string &, const string &, const string &);

void transpose(vector<vector<double>>, vector<vector<double>> &);

double cosine(vector<double>&, vector<double>&);

double maxPooling(matrix<double>&, pair<int, int>, pair<int, int>);

#endif
