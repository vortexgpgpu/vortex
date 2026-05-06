#define CL_TARGET_OPENCL_VERSION 120
#include <CL/opencl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_ERR(err, msg)                             \
  if (err != CL_SUCCESS) {                              \
    fprintf(stderr, "%s (Error code: %d)\n", msg, err); \
    return -1;                                          \
  }

int verify_result(const char *test_name, int *result, int *expected, int size) {
  for (int i = 0; i < size; i++) {
    if (result[i] != expected[i]) {
      printf("[%s] FAILED at index %d: expected %d, got %d\n", test_name, i, expected[i], result[i]);
      return -1;
    }
  }
  printf("[%s] PASSED\n", test_name);
  return 0;
}

int test_basic_copy(cl_context context, cl_command_queue queue) {
  cl_int err;
  int src_data[] = {10, 20, 30, 40, 50};
  int dst_data[5] = {0};
  size_t size = sizeof(src_data);

  cl_mem bufSrc = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, src_data, &err);
  CHECK_ERR(err, "Basic: Failed to create src buffer");
  cl_mem bufDst = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &err);
  CHECK_ERR(err, "Basic: Failed to create dst buffer");

  err = clEnqueueCopyBuffer(queue, bufSrc, bufDst, 0, 0, size, 0, NULL, NULL);
  CHECK_ERR(err, "Basic: Failed to enqueue copy");

  err = clEnqueueReadBuffer(queue, bufDst, CL_TRUE, 0, size, dst_data, 0, NULL, NULL);
  CHECK_ERR(err, "Basic: Failed to read buffer");

  clReleaseMemObject(bufSrc);
  clReleaseMemObject(bufDst);

  return verify_result("Basic Copy", dst_data, src_data, 5);
}

int test_offset_copy(cl_context context, cl_command_queue queue) {
  cl_int err;
  int src_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  int dst_data[8] = {0};

  size_t total_size = sizeof(src_data);
  size_t copy_count = 3;
  size_t copy_size = copy_count * sizeof(int);
  size_t src_offset = 2 * sizeof(int);
  size_t dst_offset = 4 * sizeof(int);

  cl_mem bufSrc = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, total_size, src_data, &err);
  CHECK_ERR(err, "Offset: Failed to create src buffer");
  cl_mem bufDst = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, total_size, dst_data, &err);
  CHECK_ERR(err, "Offset: Failed to create dst buffer");

  err = clEnqueueCopyBuffer(queue, bufSrc, bufDst, src_offset, dst_offset, copy_size, 0, NULL, NULL);
  CHECK_ERR(err, "Offset: Failed to enqueue copy");

  err = clEnqueueReadBuffer(queue, bufDst, CL_TRUE, 0, total_size, dst_data, 0, NULL, NULL);
  CHECK_ERR(err, "Offset: Failed to read buffer");

  int expected[] = {0, 0, 0, 0, 3, 4, 5, 0};

  clReleaseMemObject(bufSrc);
  clReleaseMemObject(bufDst);

  return verify_result("Offset Copy", dst_data, expected, 8);
}

int test_self_copy_no_overlap(cl_context context, cl_command_queue queue) {
  cl_int err;
  // [10, 20, 30, 0, 0, 0]
  int data[] = {10, 20, 30, 0, 0, 0};
  size_t size = sizeof(data);

  cl_mem buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, data, &err);
  CHECK_ERR(err, "Self: Failed to create buffer");

  err = clEnqueueCopyBuffer(queue, buf, buf, 0, 3 * sizeof(int), 3 * sizeof(int), 0, NULL, NULL);
  CHECK_ERR(err, "Self: Failed to enqueue copy");

  err = clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, size, data, 0, NULL, NULL);
  CHECK_ERR(err, "Self: Failed to read buffer");

  int expected[] = {10, 20, 30, 10, 20, 30};

  clReleaseMemObject(buf);

  return verify_result("Self Copy (No Overlap)", data, expected, 6);
}

int test_large_copy(cl_context context, cl_command_queue queue) {
  cl_int err;
  const int count = 1024 * 10;
  size_t size = count * sizeof(int);

  int *src_data = (int *)malloc(size);
  int *dst_data = (int *)malloc(size);

  for (int i = 0; i < count; ++i) {
    src_data[i] = i;
    dst_data[i] = 0;
  }

  cl_mem bufSrc = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, src_data, &err);
  CHECK_ERR(err, "Large: Failed to create src buffer");
  cl_mem bufDst = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &err);
  CHECK_ERR(err, "Large: Failed to create dst buffer");

  err = clEnqueueCopyBuffer(queue, bufSrc, bufDst, 0, 0, size, 0, NULL, NULL);
  CHECK_ERR(err, "Large: Failed to enqueue copy");

  err = clEnqueueReadBuffer(queue, bufDst, CL_TRUE, 0, size, dst_data, 0, NULL, NULL);
  CHECK_ERR(err, "Large: Failed to read buffer");

  int res = verify_result("Large Copy", dst_data, src_data, count);

  clReleaseMemObject(bufSrc);
  clReleaseMemObject(bufDst);
  free(src_data);
  free(dst_data);

  return res;
}

int main() {
  cl_int err;
  cl_uint num_platforms = 0;
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (err != CL_SUCCESS || num_platforms == 0) {
    fprintf(stderr, "No OpenCL platform\n");
    return -1;
  }
  cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
  clGetPlatformIDs(num_platforms, platforms, NULL);

  cl_uint num_devices = 0;
  err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
  if (err != CL_SUCCESS || num_devices == 0) {
    fprintf(stderr, "No OpenCL device\n");
    free(platforms);
    return -1;
  }

  cl_device_id *devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
  clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

  cl_context context = clCreateContext(NULL, 1, &devices[0], NULL, NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Failed to create context\n");
    return -1;
  }

  cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Failed to create queue\n");
    return -1;
  }

  printf("Running OpenCL CopyBuffer Tests...\n");
  printf("----------------------------------\n");

  int failures = 0;
  if (test_basic_copy(context, queue) != 0)
    failures++;
  if (test_offset_copy(context, queue) != 0)
    failures++;
  if (test_self_copy_no_overlap(context, queue) != 0)
    failures++;
  if (test_large_copy(context, queue) != 0)
    failures++;

  printf("----------------------------------\n");
  if (failures == 0) {
    printf("ALL TESTS PASSED\n");
  } else {
    printf("%d TESTS FAILED\n", failures);
  }

  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  free(devices);
  free(platforms);

  return failures;
}