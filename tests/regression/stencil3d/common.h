#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE float
#endif

typedef struct
{
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint32_t size;
    uint32_t block_size;
    uint64_t A_addr;
    uint64_t B_addr;
} kernel_arg_t;

#endif