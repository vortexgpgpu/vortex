#ifndef _FIND_ELLIPSE_KERNEL_H_
#define _FIND_ELLIPSE_KERNEL_H_


#ifdef __cplusplus
extern "C" {
#endif
extern float *GICOV_OpenCL(int grad_m, int grad_n, float *host_grad_x, float *host_grad_y);
extern float *dilate_OpenCL(int max_gicov_m, int max_gicov_n, int strel_m, int strel_n);
extern void select_device();
extern void transfer_constants(float *host_sin_angle, float *host_cos_angle, int *host_tX, int *host_tY, int strel_m, int strel_n, float *host_strel);
#ifdef __cplusplus
}
#endif


#endif
