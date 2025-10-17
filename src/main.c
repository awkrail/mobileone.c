#include <stdio.h>

#include "mat.h"
#include "image.h"

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

void forward(struct Mat * image, const char * weight_files [], const char * bias_files [], int weight_count)
{
    /**
    for (int i = 0; i < weight_count; i++)
    {
    }
    **/

    printf("hi\n");
}

void forward_stage0(struct Mat * image)
{
    const char * weight_files [] = {
        "stage0.reparam_conv.weight",
    };
    const char * bias_files [] = {
        "stage0.reparam_conv.bias",
    };
    const int weight_count = sizeof(weight_files) / sizeof(weight_files[0]);
    const int bias_count = sizeof(bias_files) / sizeof(bias_files[0]);
    if (weight_count != bias_count)
    {
        fprintf(stderr, "The number of weights and bias is different. Abort.\n");
        return;
    }
    return forward(image, weight_files, bias_files, weight_count);
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
