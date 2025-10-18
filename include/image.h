#ifndef IMAGE_H
#define IMAGE_H

int load_image(const char * filename, struct Mat * image);
void free_image(struct Mat * image);

#endif
