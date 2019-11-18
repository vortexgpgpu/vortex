
#include "../../intrinsics/vx_intrinsics.h"
#include "../../io/vx_io.h"
#include "../../tests/tests.h"
#include "../../vx_api/vx_api.h"
#include "../../fileio/fileio.h"
#include <CL/opencl.h>

// Newlib
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

#define MAX_KERNELS 1
#define KERNEL_NAME "vecadd"
#define KERNEL_FILE_NAME "vecadd.pocl"
#define SIZE 4
#define NUM_WORK_GROUPS 2

#define CL_CHECK(_expr)                                                \
   do {                                                                \
     cl_int _err = _expr;                                              \
     if (_err == CL_SUCCESS)                                           \
       break;                                                          \
     printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
   cleanup();                                                          \
     exit(-1);                                                         \
   } while (0)

#define CL_CHECK2(_expr)                                               \
   ({                                                                  \
     cl_int _err = CL_INVALID_VALUE;                                   \
     typeof(_expr) _ret = _expr;                                       \
     if (_err != CL_SUCCESS) {                                         \
       printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
     cleanup();                                                        \
       exit(-1);                                                       \
     }                                                                 \
     _ret;                                                             \
   })

typedef struct {
  const char* name;
  const void* pfn;
  uint32_t num_args;
  uint32_t num_locals;
  const uint8_t* arg_types;
  const uint32_t* local_sizes;
} kernel_info_t;

static int g_num_kernels = 0;
static kernel_info_t g_kernels [MAX_KERNELS];

#ifdef __cplusplus
extern "C" {
#endif

int _pocl_register_kernel(const char* name, const void* pfn, uint32_t num_args, uint32_t num_locals, const uint8_t* arg_types, const uint32_t* local_sizes) {
  //printf("******** _pocl_register_kernel\n");
  //printf("Name to register: %s\n", name);
  //printf("PTR of name: %x\n", name);
  if (g_num_kernels == MAX_KERNELS)
  {
    //printf("ERROR: REACHED MAX KERNELS\n");
    return -1;  
  }

  //printf("Going to register at index: %d\n", g_num_kernels);

  kernel_info_t* kernel = g_kernels + g_num_kernels++;
  kernel->name = name;
  kernel->pfn = pfn;
  kernel->num_args = num_args;
  kernel->num_locals = num_locals;
  kernel->arg_types = arg_types;
  kernel->local_sizes = local_sizes;
  //printf("New kernel name: %s\n", kernel->name);
  return 0;
}

int _pocl_query_kernel(const char* name, const void** p_pfn, uint32_t* p_num_args, uint32_t* p_num_locals, const uint8_t** p_arg_types, const uint32_t** p_local_sizes) {
  //printf("********* Inside _pocl_query_kernel\n");
  //printf("name: %s\n", name);
  //printf("g_num_kernels: %d\n", g_num_kernels);
  for (int i = 0; i < g_num_kernels; ++i) {
    //printf("Currently quering index %d\n", i);
    kernel_info_t* kernel = g_kernels + i;
    if (strcmp(kernel->name, name) != 0)
    {
      //printf("STR CMP failed! kernel->name = %s \t name: %s\n", kernel->name, name);
      continue;
    }
    //printf("!!!!!!!!!STR CMP PASSED\n");
    if (p_pfn) *p_pfn = kernel->pfn;
    if (p_num_args) *p_num_args = kernel->num_args;
    if (p_num_locals) *p_num_locals = kernel->num_locals;
    if (p_arg_types) *p_arg_types = kernel->arg_types;
    if (p_local_sizes) *p_local_sizes = kernel->local_sizes;
    return 0;
  }
  return -1;
}

#ifdef __cplusplus
}
#endif

unsigned *A = NULL;
unsigned *B = NULL;
unsigned *C = NULL;


// struct context_t {

//   unsigned num_groups[3];            // use {2, 1, 1} for vecadd

//   unsigned global_offset[3];         // use {0, 0, 0} for vecadd

//   unsigned local_size[3];            // use {2, 1, 1} for vecadd

//   unsigned char *printf_buffer;  // zero for now

//   unsigned *printf_buffer_position;  // initialized to zero

//   unsigned printf_buffer_capacity;   // zero for now

//   unsigned work_dim;                 // use ‘1’ for vecadd

// };

int main (int argc, char **argv) {  
  vx_tmc(1);

  printf("\n\n******** Fixing fileio START Native Vecadd running ********\n\n");


  FILE *f = fopen("/home/fares/Desktop/Vortex/simX/reading_data.txt", "r");
  fseek(f, 0, SEEK_END);
  int fsize = ftell(f);
  fseek(f, 0, SEEK_SET);  /* same as rewind(f); */

  char *string = (char *) malloc(fsize + 1);
  fread(string, 1, fsize, f);
  fclose(f);

  string[fsize] = 0;

  printf("%s", string);


   // FILE *fp;
   // char buff[1024];

   // fp = fopen("/home/fares/Desktop/Vortex/simX/reading_data.txt", "r");
   // // fscanf(fp, "%s %s %s %s", buff);
   // fgets(buff, 41, (FILE*)fp);
   // printf("1 : %s\n", buff );

  exit(0);

  // Allocate memories for input arrays and output arrays.  
  A = (unsigned*)malloc(sizeof(unsigned)*SIZE);
  B = (unsigned*)malloc(sizeof(unsigned)*SIZE);
  C = (unsigned*)malloc(sizeof(unsigned)*SIZE);


  const void    * p_pfn;
        uint32_t  p_num_args;
        uint32_t  p_num_locals;
  const uint8_t * p_arg_types;
  const uint32_t* p_local_sizes;

  int found = _pocl_query_kernel(KERNEL_NAME, &p_pfn, &p_num_args, &p_num_locals, &p_arg_types, &p_local_sizes);

  if (found == -1)
  {
    printf("_pocl_query_kernel did not find kernel!\n");
    return 1;
  }
  
  printf("p_pfn       : %x\n", p_pfn);
  printf("p_num_args  : %d\n", p_num_args);
  printf("p_num_locals: %d\n", p_num_locals);

  int i;
  // Initialize values for array members.  
  for (i=0; i<SIZE; ++i)
  {
    A[i] = i*2+0;
    B[i] = i*2+1;
  }


  // CTX initialization
  printf("ctx size: %d\n", sizeof(context_t));
  context_t * ctx = (context_t *) malloc(sizeof(context_t));
  memset(ctx, 0, sizeof(context_t));
  ctx->num_groups[0] = 4;
  ctx->num_groups[1] = 1;
  ctx->num_groups[2] = 1;

  ctx->global_offset[0] = 0;
  ctx->global_offset[1] = 0;
  ctx->global_offset[2] = 0;

  ctx->local_size[0] = 1;
  ctx->local_size[1] = 1;
  ctx->local_size[2] = 1;

  ctx->printf_buffer          = NULL;
  ctx->printf_buffer_position = 0;
  ctx->printf_buffer_capacity = 0;
  ctx->work_dim = 1;

  // Arguments initialization
  void **args = (void **)malloc (sizeof(void *) * (p_num_args + p_num_locals));
  args[0] = &A;
  args[1] = &B;
  args[2] = &C;

  printf("A address: %x\n", A);
  printf("B address: %x\n", B);
  printf("C address: %x\n", C);
  printf("args address: %x\n", args);
  printf("CTX address: %x\n", ctx);

  pocl_spawn(ctx, p_pfn, (void *) args);

  // Testing
  for (i = 0; i < SIZE; ++i)
  {
    printf("Index A[%d]=%d\tB[%d]=%d\tC[%d]=%d\n",i,A[i], i, B[i], i, C[i]);
  }

  for (i=0; i<SIZE; ++i)
  {
    if (C[i] != (A[i] + B[i]))
    {
      printf("Failed!\n");
      break;
    }
  }

  if (i == SIZE)
  {
    printf("Ok!\n");
  }


  return 0;
}
