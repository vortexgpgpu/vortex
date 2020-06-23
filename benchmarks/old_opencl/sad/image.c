/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "file.h"
#include "image.h"

struct image_i16 *
load_image(char *filename)
{
  FILE *infile;
  short *data;
  int w;
  int h;

  infile = fopen(filename, "r");

  if (!infile)
    {
      fprintf(stderr, "Cannot find file '%s'\n", filename);
      exit(-1);
    }

  /* Read image dimensions */
  w = read16u(infile);
  h = read16u(infile);

  /* Read image contents */
  data = (short *)malloc(w * h * sizeof(short));
  fread(data, sizeof(short), w * h, infile);

  fclose(infile);

  /* Create the return data structure */
  {
    struct image_i16 *ret =
      (struct image_i16 *)malloc(sizeof(struct image_i16));
    ret->width = w;
    ret->height = h;
    ret->data = data;
    return ret;
  }
}

void
free_image(struct image_i16 *img)
{
  free(img->data);
  free(img);
}
