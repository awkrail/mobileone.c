#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#include "mat.h"
#include "image.h"
#include "conv2d.h"

void dump(struct Mat * mat)
{
    int num_elem = mat->height * mat->width * mat->channel;
    FILE * fp = fopen("vector.txt", "w");
    if (!fp)
    {
        fprintf(stderr, "failed to open.\n");
        return;
    }
    for (int i = 0; i < num_elem; i++)
    {
        fprintf(fp, "%f\n", mat->data[i]);
    }
    fclose(fp);
}

void replace_mat(struct Mat * dst, struct Mat * src)
{
    free(dst->data);
    dst->height  = src->height;
    dst->width   = src->width;
    dst->channel = src->channel;
    dst->data = src->data;
    src->data = NULL;
}

int forward_layer(struct Mat * input, const char ** weight_files, const char ** bias_files,
                  int * in_channels, int * out_channels, int * kernels, int * strides, int * paddings,
                  int * groups, int num_layers)
{
    struct Mat out;
    int res;
    for (int i = 0; i < num_layers; i++)
    {
        res = forward_conv2d(weight_files[i], bias_files[i], in_channels[i], out_channels[i],
                             kernels[i], strides[i], paddings[i], groups[i], input, &out);
        if (res != 0)
        {
            // TODO: release intermediate vectors
            return res;
        }
        replace_mat(input, &out);
    }
    dump(input);
    return res;
}

int forward(struct Mat * image)
{
    const char * weight_files [] = {
        // stage 0 & 1
        "weights/stage0.reparam_conv.weight.bin",
        "weights/stage1.0.reparam_conv.weight.bin",
        "weights/stage1.1.reparam_conv.weight.bin",
        "weights/stage1.2.reparam_conv.weight.bin",
        "weights/stage1.3.reparam_conv.weight.bin",
        // stage 2
        "weights/stage2.0.reparam_conv.weight.bin",
        "weights/stage2.1.reparam_conv.weight.bin",
        "weights/stage2.2.reparam_conv.weight.bin",
        "weights/stage2.3.reparam_conv.weight.bin",
        "weights/stage2.4.reparam_conv.weight.bin",
        "weights/stage2.5.reparam_conv.weight.bin",
        "weights/stage2.6.reparam_conv.weight.bin",
        "weights/stage2.7.reparam_conv.weight.bin",
        "weights/stage2.8.reparam_conv.weight.bin",
        "weights/stage2.9.reparam_conv.weight.bin",
        "weights/stage2.10.reparam_conv.weight.bin",
        "weights/stage2.11.reparam_conv.weight.bin",
        "weights/stage2.12.reparam_conv.weight.bin",
        "weights/stage2.13.reparam_conv.weight.bin",
        "weights/stage2.14.reparam_conv.weight.bin",
        "weights/stage2.15.reparam_conv.weight.bin",

    };

    const char * bias_files [] = {
        // stage 0 & 1
        "weights/stage0.reparam_conv.bias.bin",
        "weights/stage1.0.reparam_conv.bias.bin",
        "weights/stage1.1.reparam_conv.bias.bin",
        "weights/stage1.2.reparam_conv.bias.bin",
        "weights/stage1.3.reparam_conv.bias.bin",
        // stage 2
        "weights/stage2.0.reparam_conv.bias.bin",
        "weights/stage2.1.reparam_conv.bias.bin",
        "weights/stage2.2.reparam_conv.bias.bin",
        "weights/stage2.3.reparam_conv.bias.bin",
        "weights/stage2.4.reparam_conv.bias.bin",
        "weights/stage2.5.reparam_conv.bias.bin",
        "weights/stage2.6.reparam_conv.bias.bin",
        "weights/stage2.7.reparam_conv.bias.bin",
        "weights/stage2.8.reparam_conv.bias.bin",
        "weights/stage2.9.reparam_conv.bias.bin",
        "weights/stage2.10.reparam_conv.bias.bin",
        "weights/stage2.11.reparam_conv.bias.bin",
        "weights/stage2.12.reparam_conv.bias.bin",
        "weights/stage2.13.reparam_conv.bias.bin",
        "weights/stage2.14.reparam_conv.bias.bin",
        "weights/stage2.15.reparam_conv.bias.bin",

        // stage 3
    };

    int in_channels [] = {
        // stage 0 & 1
        3,
        48,
        48,
        48,
        48,
        // stage 2
        48,
        48,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
    };

    int out_channels [] = {
        // stage 0 & 1
        48,
        48,
        48,
        48,
        48,
        // stage 2
        48,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
    };

    int kernels [] = {
        // stage 0 & 1
        3,
        3,
        1,
        3,
        1,
        // stage 2
        3,
        1,
        3,
        1,
        3,
        1,
        3,
        1,
        3,
        1,
        3,
        1,
        3,
        1,
        3,
        1,
    };

    int strides [] = {
        // stage 0 & 1
        2,
        2,
        1,
        1,
        1,
        // stage 2
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    };

    int paddings [] = {
        // stage 0 & 1
        1,
        1,
        0,
        1,
        0,
        // stage 2
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
    };

    int groups [] = {
        // stage 0 & 1
        1,
        48,
        1,
        48,
        1,
        // stage 2
        48,
        1,
        128,
        1,
        128,
        1,
        128,
        1,
        128,
        1,
        128,
        1,
        128,
        1,
        128,
        1,
    };

    int num_layer = sizeof(weight_files) / sizeof(weight_files[0]);
    return forward_layer(image, weight_files, bias_files, in_channels, out_channels, kernels, strides, paddings, groups, num_layer);
}

int main(int argc, char * argv [])
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: ./mobileone [input_image_file].\n");
        return 1;
    }

    struct Mat image;
    if (load_image(argv[1], &image) != 0)
    {
        fprintf(stderr, "Error in load_image. Abort.\n");
        return 1;
    }

    if (forward(&image) != 0)
    {
        fprintf(stderr, "Error in forward_stage0. Abort.\n");
        return 1;
    }

    free_image(&image);
    return 0;
}
