#ifndef CONV2D_H
#define CONV2D_H

#include "mat.h"

struct Conv2D
{
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    float * weights;
    float * bias;
};

struct Mat forward_conv2d(const char * weight_file, const char * bias_file, struct Mat * input);

#endif
