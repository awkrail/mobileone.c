#include <stdio.h>
#include <stdlib.h>

#include "linear.h"
#include "load.h"

void release_linear_layer(struct Linear * linear_layer)
{
    if (linear_layer->weights)
    {
        free(linear_layer->weights);
        linear_layer->weights = NULL;
    }
    
    if (linear_layer->bias)
    {
        free(linear_layer->bias);
        linear_layer->bias = NULL;
    }
}

int linear(const char * weight_file, const char * bias_file, 
           int in_features, int out_features, struct Mat * input, struct Mat * output)
{
    struct Linear linear_layer;
    linear_layer.in_features = in_features;
    linear_layer.out_features = out_features;
    linear_layer.weights = load(weight_file);
    linear_layer.bias = load(bias_file);

    if (!linear_layer.weights || !linear_layer.bias)
    {
        fprintf(stderr, "Failed to load weights or bias.\n");
        release_linear_layer(&linear_layer);
        return 1;
    }

    output->channel = 1;
    output->height = 1;
    output->width = out_features;
    output->data = (float *)malloc(out_features * sizeof(float));
    if (!output->data)
    {
        fprintf(stderr, "Failed to allocate linear output buffer.\n");
        release_linear_layer(&linear_layer);
        return 1;
    }

    for (int o = 0; o < out_features; o++)
    {
        float sum = linear_layer.bias[o];
        for (int i = 0; i < in_features; i++)
        {
            int w_idx = o * in_features + i;
            sum += linear_layer.weights[w_idx] * input->data[i];
        }
        output->data[o] = sum;
    }

    release_linear_layer(&linear_layer);
    return 0;
}
