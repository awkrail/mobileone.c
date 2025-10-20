#ifndef CONV2D_H
#define CONV2D_H

#include "mat.h"

struct Conv2D
{
    int in_channel;
    int out_channel;
    int kernel_size;
    int stride;
    int padding;
    int group;
    float * weights;
    float * bias;
};

int forward_conv2d(const char * weight_file, const char * bias_file, int in_channel, int out_channel,
                   int kernel, int stride, int padding, int group, struct Mat * input, struct Mat * output);

#endif
