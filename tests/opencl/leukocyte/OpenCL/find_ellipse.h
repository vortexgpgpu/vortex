#ifndef FIND_ELLIPSE_H
#define FIND_ELLIPSE_H

#include <CL/cl.h>
#include "avilib.h"
#include "../meschach_lib/matrix.h"
#include "misc_math.h"
#include <math.h>
#include <stdlib.h>

// Defines the region in the video frame containing the blood vessel
#define TOP 110
#define BOTTOM 328

extern long long get_time();

// Global variables used by OpenCL functions
extern cl_context context;
extern cl_command_queue command_queue;
extern cl_device_id device;

extern MAT * get_frame(avi_t *cell_file, int frame_num, int cropped, int scaled);
extern MAT * chop_flip_image(unsigned char *image, int height, int width, int top, int bottom, int left, int right, int scaled);
extern MAT * GICOV(MAT * grad_x, MAT * grad_y);
extern MAT * dilate(MAT * img_in);
extern MAT * linear_interp2(MAT * m, VEC * X, VEC * Y);
extern MAT * TMatrix(unsigned int N, unsigned int M);
extern VEC * getsampling(MAT * m, int ns);
extern VEC * getfdriv(MAT * m, int ns);

extern void choose_GPU();
extern void compute_constants();
extern void uniformseg(VEC * cellx_row, VEC * celly_row, MAT * x, MAT * y);
extern void splineenergyform01(MAT * Cx, MAT * Cy, MAT * Ix, MAT * Iy, int ns, double delta, double dt, int typeofcell);

extern float * structuring_element(int radius);

extern double m_min(MAT * m);
extern double m_max(MAT * m);

#endif
