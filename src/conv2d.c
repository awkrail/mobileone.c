#include <stdio.h>
#include <stdlib.h>

#include "load.h"
#include "conv2d.h"

void release_conv2d_layer(struct Conv2D * conv2d_layer)
{
    if (conv2d_layer->weights)
    {
        free(conv2d_layer->weights);
        conv2d_layer->weights = NULL;
    }
    if (conv2d_layer->bias)
    {
        free(conv2d_layer->bias);
        conv2d_layer->bias = NULL;
    }
}

static inline float relu(float val)
{
    return val > 0 ? val : 0.0f;
}

int forward_conv2d(const char *weight_file, const char *bias_file,
                   int in_channel, int out_channel, int kernel, int stride,
                   int padding, int group, struct Mat *input, struct Mat *output)
{
    struct Conv2D conv2d_layer;
    conv2d_layer.in_channel = in_channel;
    conv2d_layer.out_channel = out_channel;
    conv2d_layer.kernel_size = kernel;
    conv2d_layer.stride = stride;
    conv2d_layer.padding = padding;
    conv2d_layer.group = group;
    conv2d_layer.weights = load(weight_file);
    conv2d_layer.bias = load(bias_file);

    if (!conv2d_layer.weights || !conv2d_layer.bias)
    {
        fprintf(stderr, "Failed to load weight or bias file.\n");
        release_conv2d_layer(&conv2d_layer);
        return 1;
    }

    if (in_channel % group != 0 || out_channel % group != 0)
    {
        fprintf(stderr, "Error: in_channel and out_channel must be divisible by group.\n");
        release_conv2d_layer(&conv2d_layer);
        return 1;
    }

    const int output_height = (input->height + 2 * padding - kernel) / stride + 1;
    const int output_width  = (input->width + 2 * padding - kernel) / stride + 1;
    const int in_per_group  = in_channel / group;
    const int out_per_group = out_channel / group;

    output->height = output_height;
    output->width  = output_width;
    output->channel = out_channel;
    output->data = (float *)malloc(output_height * output_width * out_channel * sizeof(float));
    if (!output->data)
    {
        fprintf(stderr, "Failed to allocate output buffer.\n");
        release_conv2d_layer(&conv2d_layer);
        return 1;
    }

    for (int g = 0; g < group; g++)
    {
        int in_start  = g * in_per_group;
        int out_start = g * out_per_group;
        for (int oc = 0; oc < out_per_group; oc++)
        {
            int oc_global = out_start + oc;

            for (int oh = 0; oh < output_height; oh++)
            {
                for (int ow = 0; ow < output_width; ow++)
                {
                    float val = 0.0f;
                    for (int ic = 0; ic < in_per_group; ic++)
                    {
                        int ic_global = in_start + ic;

                        for (int kh = 0; kh < kernel; kh++)
                        {
                            for (int kw = 0; kw < kernel; kw++)
                            {
                                int in_y = oh * stride + kh - padding;
                                int in_x = ow * stride + kw - padding;

                                if (in_y < 0 || in_x < 0 ||
                                    in_y >= input->height || in_x >= input->width)
                                    continue;

                                int data_index =
                                    ic_global * input->height * input->width +
                                    in_y * input->width + in_x;

                                int weight_index =
                                    (oc_global * in_per_group * kernel * kernel) +
                                    (ic * kernel * kernel) +
                                    (kh * kernel + kw);

                                val += input->data[data_index] * conv2d_layer.weights[weight_index];
                            }
                        }
                    }

                    int output_index = oc_global * output_height * output_width +
                                       oh * output_width + ow;
                    output->data[output_index] = relu(val + conv2d_layer.bias[oc_global]);
                }
            }
        }
    }
    release_conv2d_layer(&conv2d_layer);
    return 0;
}
