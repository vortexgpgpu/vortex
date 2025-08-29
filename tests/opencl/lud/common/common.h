#ifndef _COMMON_H
#define _COMMON_H

#include <time.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif



#define GET_RAND_FP ( (float)rand() /   \
                     ((float)(RAND_MAX)+(float)(1)) )

#define MIN(i,j) ((i)<(j) ? (i) : (j))

typedef enum _FUNC_RETURN_CODE {
    RET_SUCCESS,
    RET_FAILURE
}func_ret_t;

typedef struct __stopwatch_t{
    struct timeval begin;
    struct timeval end;
}stopwatch;

void 
stopwatch_start(stopwatch *sw);

void 
stopwatch_stop (stopwatch *sw);

double 
get_interval_by_sec(stopwatch *sw);

int 
get_interval_by_usec(stopwatch *sw);

func_ret_t
create_matrix_from_file(float **mp, const char *filename, int *size_p);

func_ret_t
create_matrix_from_random(float **mp, int size);

func_ret_t
create_matrix(float **mp, int size);

func_ret_t
lud_verify(float *m, float *lu, int size);

void
matrix_multiply(float *inputa, float *inputb, float *output, int size);

void
matrix_duplicate(float *src, float **dst, int matrix_dim);

void
print_matrix(float *mm, int matrix_dim);

#ifdef __cplusplus
}
#endif

#endif
