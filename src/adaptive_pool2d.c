#include <stdio.h>
#include <stdlib.h>

#include "adaptive_pool2d.h"

int adaptive_avg_pool2d(struct Mat * input, struct Mat * output)
{
    output->height = 1;
    output->width = 1;
    output->channel = input->channel;
    output->data = (float*)malloc(input->channel * sizeof(float));
    if (!output->data)
    {
        fprintf(stderr, "Failed to allocate memory.\n");
        return 1;
    }

    for (int c = 0; c < input->channel; c++)
    {
        float sum = 0.0f;
        for (int h = 0; h < input->height; h++)
        {
            for (int w = 0; w < input->width; w++)
            {
                int idx = c * input->height * input->width + h * input->width + w;
                sum += input->data[idx];
            }
        }
        output->data[c] = sum / (input->height * input->width);
    }

    return 0;
}
