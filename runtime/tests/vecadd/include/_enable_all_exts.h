/* Enable all extensions known to pocl, which a device supports.
 * This is required at the start of include/_kernel.h for prototypes,
 * then at kernel lib compilation phase (because _kernel.h disables
 * everything at the end).
 */

/* OpenCL 1.0-only extensions */

#if (__OPENCL_C_VERSION__ < 110)

#ifdef cl_khr_global_int32_base_atomics
#  pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#endif

#ifdef cl_khr_global_int32_extended_atomics
#  pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#endif

#ifdef cl_khr_local_int32_base_atomics
#  pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#endif

#ifdef cl_khr_local_int32_extended_atomics
#  pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#endif

#ifdef cl_khr_byte_addressable_store
#  pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#endif

#endif


/* all versions */
#ifdef cl_khr_fp16
#  pragma OPENCL EXTENSION cl_khr_fp16: enable
#endif

#ifdef cl_khr_fp64
#  pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

#ifdef cl_khr_int64_base_atomics
#  pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif

#ifdef cl_khr_int64_extended_atomics
#  pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
#endif

#if (__clang_major__ > 4)

#ifdef cl_khr_3d_image_writes
#  pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
#endif

#endif

