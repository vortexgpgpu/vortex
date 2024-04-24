extern cl_device_id device_id;
extern cl_context context;

#define CL_CHECK(_expr)                                                \
   do {                                                                \
     cl_int _err = _expr;                                              \
     if (_err == CL_SUCCESS)                                           \
       break;                                                          \
     printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
     exit(-1);                                                         \
   } while (0)

#define CL_CHECK2(_expr)                                               \
   ({                                                                  \
     cl_int _err = CL_INVALID_VALUE;                                   \
     decltype(_expr) _ret = _expr;                                     \
     if (_err != CL_SUCCESS) {                                         \
       printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
       exit(-1);                                                       \
     }                                                                 \
     _ret;                                                             \
   })


#define TEST(func)                                                     \
  ({                                                                   \
    char function_name[] = #func;                                      \
    printf("Running %s.\n",function_name);                             \
    uint32_t result = func();                                          \
    if (!result) printf("PASSED.\n");                                  \
    else printf("FAILED with %d errors.\n", result);                   \
    errors += result;                                                  \
  })

#include "test_readnpixels.c"
#include "test_perspective_div.c"
#include "test_color_kernel.c"
#include "test_viewport_trans.c"
