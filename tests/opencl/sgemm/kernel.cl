__kernel void sgemm (__global const float *A,
	                   __global const float *B,
	                   __global float *C, 
                     int N)
{
  // Thread identifiers
  const int r = get_global_id(0); // Row ID
  const int c = get_global_id(1); // Col ID

  // Compute a single element (loop a K)
  float acc = 0.0f;
  for (int k = 0; k < N; k++) {
    acc += A[k * N + r] * B[c * N + k];
  }

  // Store the result
  C[c * N + r] = acc;
}
