#include <stdio.h>
#include <stdbool.h>

#include "mat.h"
#include "image.h"

void forward(struct Mat * input, const char * weight_files [], const char * bias_files [], int weight_count)
{
    for (int i = 0; i < weight_count; i++)
    {
        //struct Mat out = forward_conv2d(weight_files[i], bias_files[i], input);
    }
}

static bool is_config_weight_num_equal(int lens [], int lens_count)
{
    bool all_eq = true;
    for (int i = 1; i < lens_count; i++)
    {
        if (lens[i] != lens[0])
        {
            all_eq = false;
            break;
        }
    }
    return true;
}

void forward_stage0(struct Mat * image)
{
    /* Stage0 weights & configs */
    const char * weight_files [] = {
        "stage0.reparam_conv.weight",
    };

    const char * bias_files [] = {
        "stage0.reparam_conv.bias",
    };

    int in_channels [] = {
        3,
    };

    int out_channels [] = {
        3,
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

    int lens [] = {
        sizeof(weight_files) / sizeof(weight_files[0]),
        sizeof(bias_files) / sizeof(bias_files[0]),
        sizeof(in_channels) / sizeof(in_channels[0]),
        sizeof(out_channels) / sizeof(out_channels[0]),
        sizeof(kernels) / sizeof(kernels[0]),
        sizeof(strides) / sizeof(strides[0]),
        sizeof(paddings) / sizeof(paddings[0]),
    };

    int lens_count = sizeof(lens) / sizeof(lens[0]);
    if (!is_config_weight_num_equal(lens, lens_count))
    {
        fprintf(stderr, "The number of weights and bias is different. Abort.\n");
        return;
    }

    return forward(image, weight_files, bias_files, lens[0]);
}

int main(int argc, char * argv [])
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: ./mobileone [input_image_file].\n");
        return 1;
    }

    /* load image */
    struct Mat image = load_image(argv[1]);
    forward_stage0(&image);
    free_image(&image);

    return 0;
}
