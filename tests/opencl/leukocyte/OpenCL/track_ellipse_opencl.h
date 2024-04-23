#ifndef _TRACK_ELLIPSE_KERNEL_H_
#define _TRACK_ELLIPSE_KERNEL_H_

#include "../meschach_lib/matrix.h"

#ifdef __cplusplus
extern "C" {
#endif
extern void IMGVF_OpenCL_init(MAT **I, int Nc);
extern void IMGVF_OpenCL_cleanup(MAT **IMGVF_out, int Nc);
extern void IMGVF_OpenCL(MAT **I, MAT **IMGVF, double vx, double vy, double e, int max_iterations, double cutoff, int Nc);
#ifdef __cplusplus
}
#endif


#endif
