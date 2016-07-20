//coding:utf-8
/*
Program: Utils
Description: 
Shanbo Cheng: cshanbo@gmail.com
Date: 2016-07-20 09:30:37
Last modified: 2016-07-20 09:30:37
GCC version: 4.7.3
*/
#include <iostream>
#include <cstdlib>
using namespace std;

double randRange(double fMin,  double fMax) {
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}
