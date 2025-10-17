#include <stdio.h>

#include "load.h"
#include "conv2d.h"

int forward_conv2d(const char * weight_file, const char * bias_file,
                   int in_channel, int out_channel, int kernel, int stride,
                   int padding, struct Mat * input, struct Mat * output)  
{
    struct Conv2D conv2d_layer;
    conv2d_layer.in_channel = in_channel;
    conv2d_layer.out_channel = out_channel;
    conv2d_layer.kernel_size = kernel;
    conv2d_layer.stride = stride;
    conv2d_layer.padding = padding;
    conv2d_layer.weights = load(weight_file);
    conv2d_layer.bias = load(bias_file);

    if (!conv2d_layer.weights)
    {
        fprintf(stderr, "Failed to load weight file.\n");
        return 1;
    }

    if (!conv2d_layer.bias)
    {
        fprintf(stderr, "Failed to load bias file.\n");
        return 1;
    }

    /* forward Conv2D */



    return 0;
}
