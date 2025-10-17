#include <stdio.h>
#include <stdlib.h>

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

float * load(const char * filename, size_t * count_out)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return NULL;
    }

    if (fseek(fp, 0, SEEK_END) != 0)
    {
        fprintf(stderr, "Fseek error, file-size maybe 0?\n");
        fclose(fp);
        return NULL;
    }

    long size = ftell(fp);
    if (size < 0)
    {
        fprintf(stderr, "Ftell error\n");
        fclose(fp);
        return NULL;
    }

    int count = size / sizeof(float);
    float * data = (float *)malloc(count * sizeof(float));
    if (!data)
    {
        fprintf(stderr, "Failed to allocate memory.\n");
        fclose(fp);
        return NULL;
    }

    size_t read_count = fread(data, sizeof(float), count, fp);
    fclose(fp);

    if (read_count != count)
    {
        fprintf(stderr, "Error: file size mismatch or read error.\n");
        return NULL;
    }
    
    if (count_out)
        *count_out = count;

    return data;
}



int main(int argc, char * argv [])
{
    printf("hello, world\n");
    return 0;
}
