#ifndef LINEAR_H
#define LINEAR_H

#include "mat.h"

struct Linear
{
    int in_features;
    int out_features;
    float * weights;
    float * bias;
};

int linear(const char * weight_file, const char * bias_file, 
           int in_features, int out_features, struct Mat * input, struct Mat * output);
void release_linear_layer(struct Linear * linear_layer);

#endif
