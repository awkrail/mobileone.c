#include "load.h"
#include "mat.h"
#include "image.h"

#include <stdlib.h>

struct Mat load_image(const char * filename)
{
    int height = 224;
    int width = 224;
    int channel = 3;
    float * image_data = load(filename);
    struct Mat input = { height, width, channel, image_data };
    return input;
}

void free_image(struct Mat * image)
{
    free(image->data);
    image->data = NULL;
}
