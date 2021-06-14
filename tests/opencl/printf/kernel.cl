__kernel void test_printf (__global const int *A)
{
  int gid = get_global_id(0);
  int value = A[gid];
  printf("Print Test! value[%d]=%d\n", gid, value);
}