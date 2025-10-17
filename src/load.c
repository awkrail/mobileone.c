#include <stdlib.h>
#include <stdio.h>

float * load(const char * filename)
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

    rewind(fp);

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
    
    return data;
}
