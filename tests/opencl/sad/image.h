/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

struct image_i16
{
  int width;
  int height;
  short *data;
};

#ifdef __cplusplus
extern "C" {
#endif

struct image_i16 * load_image(char *filename);
void free_image(struct image_i16 *);

#ifdef __cplusplus
}
#endif
