#include "load.h"
#include "mat.h"
#include "image.h"

#include <stdio.h>
#include <stdlib.h>

int load_image(const char * filename, struct Mat * image)
{
    image->height = 224;
    image->width = 224;
    image->channel = 3;
    image->data = load(filename);
    if (!image->data)
    {
        fprintf(stderr, "Failed to load image: %s\n", filename);
        return 1;
    }
    return 0;
}

void free_image(struct Mat * image)
{
    if (image->data)
    {
        free(image->data);
        image->data = NULL;
    }
}
