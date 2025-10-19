#include <stdio.h>
#include <stdbool.h>

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


int forward(struct Mat * input, const char ** weight_files, const char ** bias_files,
            int * in_channels, int * out_channels, int * kernels, int * strides, int * paddings, int num_layers)
{
    struct Mat out;
    int res;
    for (int i = 0; i < num_layers; i++)
    {
        res = forward_conv2d(weight_files[i], bias_files[i], in_channels[i], out_channels[i],
                             kernels[i], strides[i], paddings[i], input, &out);


    }
    return res;
}

int forward_stage0(struct Mat * image)
{
    /* Stage0 weights & configs */
    const char * weight_files [] = {
        "weights/stage0.reparam_conv.weight.bin",
    };

    const char * bias_files [] = {
        "weights/stage0.reparam_conv.bias.bin",
    };

    int in_channels [] = {
        3,
    };

    int out_channels [] = {
        48,
    };

    int kernels [] = {
        3,
    };

    int strides [] = {
        2,
    };

    int paddings [] = {
        1,
    };

    int num_layer = sizeof(weight_files) / sizeof(weight_files[0]);
    return forward(image, weight_files, bias_files, in_channels, out_channels, kernels, strides, paddings, num_layer);
}

int main(int argc, char * argv [])
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: ./mobileone [input_image_file].\n");
        return 1;
    }

    /* load image */
    struct Mat image;
    if (load_image(argv[1], &image) != 0)
    {
        fprintf(stderr, "Error in load_image. Abort.\n");
        return 1;
    }

    /* forward stage 0 */
    if (forward_stage0(&image) != 0)
    {
        fprintf(stderr, "Error in forward_stage0. Abort.\n");
        return 1;
    }

    free_image(&image);
    return 0;
}
