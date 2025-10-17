#ifndef IMAGE_H
#define IMAGE_H

struct Image
{
    int height;
    int width;
    int channel;
    float * data;
};

struct Mat load_image(const char * filename);
void free_image(struct Mat * image);

#endif
