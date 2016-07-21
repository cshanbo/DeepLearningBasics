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

#include <vector>
using namespace std;

double randRange(double, double);

void dot(vector<vector<double>>&, vector<vector<double>>&, vector<vector<double>>&, vector<double> = vector<double>{});

void print(vector<vector<double>>);

int maxIndex(vector<double>& vec);

double sigmoid(double x);

#endif
