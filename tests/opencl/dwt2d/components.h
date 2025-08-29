#include <unistd.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>
//#include <stddef.h>
#ifndef _COMPONENTS_H
#define _COMPONENTS_H

/* Separate compoents of source 8bit RGB image */

template<typename T>
void rgbToComponents(T d_r, T d_g, T d_b, unsigned char * src, int width, int height);


/* Copy a 8bit source image data into a color compoment of type T */
//template<typename T>
//void bwToComponent(T *d_c, unsigned char * src, int width, int height);

#endif
