#ifndef TRACK_ELLIPSE_H
#define TRACK_ELLIPSE_H

#include "find_ellipse.h"


extern void ellipsetrack(avi_t *video, double *xc0, double *yc0, int num_centers, int R, int Np, int Nf);
extern MAT **MGVF(MAT **I, double vx, double vy, int Nc);
extern void heaviside(MAT *H, MAT *z, double v, double e);
extern void ellipseevolve(MAT *f, double *xc0, double *yc0, double *r0, double* t, int Np, double Er, double Ey);
extern double sum_m(MAT *matrix);
extern double sum_v(VEC *vector);
extern double **alloc_2d_double(int x, int y);
extern double ***alloc_3d_double(int x, int y, int z);
extern void free_2d_double(double **p);
extern void free_3d_double(double ***p);

#endif
