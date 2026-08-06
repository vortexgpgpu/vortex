// B+ tree (Rodinia b+tree) — standalone, self-checking OpenCL port for Vortex.
//
// The original benchmark reads a data file (mil.txt) and a command file. This
// port is fully self-contained: it builds a small B+ tree over a deterministic
// set of keys in-host, generates deterministic queries, runs the GPU search
// kernels, and validates the results against a serial CPU reference.
//
// Two kernels are exercised:
//   findK       — point lookup, returns the record value for each query key.
//   findRangeK  — range lookup, returns the record index of the range start
//                 and the number of records spanned by [start, end].
//
// Constraint: one work-group processes one query and its local size equals the
// tree ORDER, so ORDER is fixed at 16 to keep the work-group size <= the device
// max (NUM_WARPS*NUM_THREADS = 16). No atomics are used.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <climits>
#include <cstdint>
#include <algorithm>
#include <queue>
#include <vector>
#include <CL/opencl.h>

// Tree order == work-group size. Must be <= device max work-group size (16) and
// must match the ORDER the kernel is compiled with so the knode struct layouts
// agree byte-for-byte.
#define ORDER 16

// Record payload offset so a value is distinct from its key (surfaces any
// record-index indirection bug rather than aliasing key == value).
#define PAYLOAD_BASE 100000

#define CL_CHECK(_expr)                                                \
  do {                                                                 \
    cl_int _err = _expr;                                               \
    if (_err == CL_SUCCESS)                                            \
      break;                                                           \
    printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);    \
    cleanup();                                                         \
    exit(-1);                                                          \
  } while (0)

#define CL_CHECK2(_expr)                                               \
  ({                                                                   \
    cl_int _err = CL_INVALID_VALUE;                                    \
    decltype(_expr) _ret = _expr;                                      \
    if (_err != CL_SUCCESS) {                                          \
      printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);  \
      cleanup();                                                       \
      exit(-1);                                                        \
    }                                                                  \
    _ret;                                                              \
  })

// Host mirror of the device knode struct (identical field types/order so raw
// byte transfers are layout-compatible).
typedef struct record {
  int value;
} record;

typedef struct knode {
  int location;
  int indices[ORDER + 1];
  int keys[ORDER + 1];
  bool is_leaf;
  int num_keys;
} knode;

static int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (nullptr == filename || nullptr == data || 0 == size)
    return -1;
  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    return -1;
  }
  fseek(fp, 0, SEEK_END);
  long fsize = ftell(fp);
  rewind(fp);
  *data = (uint8_t*)malloc(fsize);
  *size = fread(*data, 1, fsize, fp);
  fclose(fp);
  return 0;
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel findK_kernel = NULL;
cl_kernel findRangeK_kernel = NULL;
cl_mem knodesD = NULL;
cl_mem recordsD = NULL;
cl_mem currKnodeD = NULL;
cl_mem offsetD = NULL;
cl_mem keysD = NULL;
cl_mem ansD = NULL;
cl_mem currKnode2D = NULL;
cl_mem offset2D = NULL;
cl_mem lastKnodeD = NULL;
cl_mem offset22D = NULL;
cl_mem startD = NULL;
cl_mem endD = NULL;
cl_mem recstartD = NULL;
cl_mem reclenD = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (findK_kernel) clReleaseKernel(findK_kernel);
  if (findRangeK_kernel) clReleaseKernel(findRangeK_kernel);
  if (program) clReleaseProgram(program);
  if (knodesD) clReleaseMemObject(knodesD);
  if (recordsD) clReleaseMemObject(recordsD);
  if (currKnodeD) clReleaseMemObject(currKnodeD);
  if (offsetD) clReleaseMemObject(offsetD);
  if (keysD) clReleaseMemObject(keysD);
  if (ansD) clReleaseMemObject(ansD);
  if (currKnode2D) clReleaseMemObject(currKnode2D);
  if (offset2D) clReleaseMemObject(offset2D);
  if (lastKnodeD) clReleaseMemObject(lastKnodeD);
  if (offset22D) clReleaseMemObject(offset22D);
  if (startD) clReleaseMemObject(startD);
  if (endD) clReleaseMemObject(endD);
  if (recstartD) clReleaseMemObject(recstartD);
  if (reclenD) clReleaseMemObject(reclenD);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

//========================================================================
//  In-host B+ tree construction (bulk-load) and flattening.
//========================================================================

// Intermediate tree node used while bulk-loading bottom-up.
struct TmpNode {
  bool is_leaf;
  int min_key;                  // smallest key in this node's subtree
  std::vector<int> keys;        // leaf: the keys; internal: separator keys
  std::vector<int> children;    // internal: indices into the node pool
  std::vector<int> recids;      // leaf: record index for each key
};

// Bulk-load a B+ tree from sorted unique keys. Leaves hold up to ORDER-1 keys;
// internal nodes fan out to at most ORDER children. Record index == sorted rank.
// Returns the pool index of the root.
static int build_tree(const std::vector<int>& K, std::vector<TmpNode>& pool) {
  const int leaf_cap = ORDER - 1;
  std::vector<int> level;

  // Leaves.
  for (int i = 0; i < (int)K.size(); i += leaf_cap) {
    TmpNode nd;
    nd.is_leaf = true;
    int end = std::min((int)K.size(), i + leaf_cap);
    for (int j = i; j < end; ++j) {
      nd.keys.push_back(K[j]);
      nd.recids.push_back(j);
    }
    nd.min_key = nd.keys.front();
    pool.push_back(nd);
    level.push_back((int)pool.size() - 1);
  }

  // Internal levels, built bottom-up until a single root remains.
  while (level.size() > 1) {
    std::vector<int> next;
    for (int i = 0; i < (int)level.size(); i += ORDER) {
      TmpNode nd;
      nd.is_leaf = false;
      int end = std::min((int)level.size(), i + ORDER);
      for (int j = i; j < end; ++j) {
        nd.children.push_back(level[j]);
        if (j > i)  // separator = smallest key of every child but the first
          nd.keys.push_back(pool[level[j]].min_key);
      }
      nd.min_key = pool[level[i]].min_key;
      pool.push_back(nd);
      next.push_back((int)pool.size() - 1);
    }
    level.swap(next);
  }
  return level[0];
}

// Height in edges from root to a leaf (== iteration count for the kernels).
static int tree_height(const std::vector<TmpNode>& pool, int root) {
  int h = 0, n = root;
  while (!pool[n].is_leaf) {
    n = pool[n].children[0];
    ++h;
  }
  return h;
}

// Flatten the tree into the BFS-ordered knode array the kernels consume (root
// at index 0), mirroring the Rodinia transform_to_cuda layout.
static void flatten_tree(std::vector<TmpNode>& pool, int root, std::vector<knode>& out) {
  std::vector<int> bfs_index(pool.size(), -1);
  std::vector<int> bfs_list;
  std::queue<int> q;
  q.push(root);
  while (!q.empty()) {
    int n = q.front();
    q.pop();
    bfs_index[n] = (int)bfs_list.size();
    bfs_list.push_back(n);
    for (int c : pool[n].children)
      q.push(c);
  }

  out.resize(bfs_list.size());
  for (int n : bfs_list) {
    const TmpNode& t = pool[n];
    knode& k = out[bfs_index[n]];
    k.location = bfs_index[n];
    k.is_leaf = t.is_leaf;
    // Pad: sentinel keys and zeroed indices.
    for (int i = 0; i <= ORDER; ++i) {
      k.keys[i] = INT_MAX;
      k.indices[i] = 0;
    }
    k.keys[0] = INT_MIN;

    if (t.is_leaf) {
      int cnt = (int)t.keys.size();
      k.num_keys = cnt + 2;
      for (int i = 0; i < cnt; ++i) {
        k.keys[i + 1] = t.keys[i];
        k.indices[i + 1] = t.recids[i];
      }
      k.indices[0] = 0;
    } else {
      int ch = (int)t.children.size();
      int seps = (int)t.keys.size();  // == ch - 1
      k.num_keys = seps + 2;
      for (int i = 0; i < seps; ++i)
        k.keys[i + 1] = t.keys[i];
      for (int i = 0; i < ch; ++i)
        k.indices[i] = bfs_index[t.children[i]];
    }
  }
}

//========================================================================
//  Serial CPU references.
//========================================================================

// findK: value of the record for key q, or -1 if absent.
static int cpu_findK(const std::vector<int>& K, const std::vector<int>& recValues, int q) {
  auto it = std::lower_bound(K.begin(), K.end(), q);
  if (it != K.end() && *it == q)
    return recValues[it - K.begin()];
  return -1;
}

//========================================================================
//  Host driver.
//========================================================================

static int num_keys = 256;       // tree size (deterministic keys)
static int num_kqueries = 32;    // number of findK point queries
static int num_rqueries = 16;    // number of findRangeK range queries
static int range_span = 8;       // max span (in ranks) of a range query

static void show_usage() {
  printf("Usage: [-n num_keys] [-k findK_queries] [-r findRangeK_queries] [-s range_span] [-h]\n");
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:r:s:h")) != -1) {
    switch (c) {
    case 'n': num_keys = atoi(optarg); break;
    case 'k': num_kqueries = atoi(optarg); break;
    case 'r': num_rqueries = atoi(optarg); break;
    case 's': range_span = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default: show_usage(); exit(-1);
    }
  }
  if (num_keys < 2 || num_kqueries < 1 || num_rqueries < 0 || range_span < 1) {
    printf("Error: invalid parameters\n");
    exit(-1);
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  // --- Deterministic keys and records. Keys are the even integers 0,2,4,...
  // (gaps let odd queries miss); record value = key + PAYLOAD_BASE. ---
  std::vector<int> K(num_keys);
  std::vector<int> recValues(num_keys);
  for (int i = 0; i < num_keys; ++i) {
    K[i] = 2 * i;
    recValues[i] = K[i] + PAYLOAD_BASE;
  }

  // --- Build and flatten the tree. ---
  std::vector<TmpNode> pool;
  int root = build_tree(K, pool);
  int height = tree_height(pool, root);
  std::vector<knode> knodes;
  flatten_tree(pool, root, knodes);
  int knodes_elem = (int)knodes.size();

  printf("B+ tree: order=%d work_group_size=%d keys=%d knodes=%d height=%d\n",
         ORDER, ORDER, num_keys, knodes_elem, height);
  printf("Queries: findK=%d findRangeK=%d range_span=%d\n",
         num_kqueries, num_rqueries, range_span);

  // --- Deterministic query generation. ---
  srand(23);
  std::vector<int> qkeys(num_kqueries);
  for (int i = 0; i < num_kqueries; ++i)
    qkeys[i] = rand() % (2 * num_keys);  // even -> hit, odd -> miss

  std::vector<int> rstart(num_rqueries), rend(num_rqueries);
  std::vector<int> rstart_rank(num_rqueries), rend_rank(num_rqueries);
  for (int i = 0; i < num_rqueries; ++i) {
    int si = rand() % num_keys;
    int ei = std::min(num_keys - 1, si + (rand() % range_span));
    rstart_rank[i] = si;
    rend_rank[i] = ei;
    rstart[i] = K[si];
    rend[i] = K[ei];
  }

  // --- OpenCL setup. ---
  cl_platform_id platform_id;
  size_t kernel_size;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));
  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
      context, 1, (const char**)&kernel_bin, &kernel_size, &_err));
  // Pin the kernel ORDER to the host ORDER so struct layouts match.
  CL_CHECK(clBuildProgram(program, 1, &device_id, "-D ORDER=" "16", NULL, NULL));
  findK_kernel = CL_CHECK2(clCreateKernel(program, "findK", &_err));
  findRangeK_kernel = CL_CHECK2(clCreateKernel(program, "findRangeK", &_err));

  // Shared tree buffers.
  knodesD = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                     sizeof(knode) * knodes_elem, NULL, &_err));
  recordsD = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      sizeof(record) * num_keys, NULL, &_err));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, knodesD, CL_TRUE, 0,
                                sizeof(knode) * knodes_elem, knodes.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, recordsD, CL_TRUE, 0,
                                sizeof(record) * num_keys, recValues.data(), 0, NULL, NULL));

  cl_long cl_height = height;
  cl_long cl_knodes_elem = knodes_elem;
  int errors = 0;

  //========================================================================
  //  findK
  //========================================================================
  {
    std::vector<cl_long> currKnode(num_kqueries, 0);
    std::vector<cl_long> offset(num_kqueries, 0);
    std::vector<record> ans(num_kqueries);
    for (int i = 0; i < num_kqueries; ++i)
      ans[i].value = -1;

    currKnodeD = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          sizeof(cl_long) * num_kqueries, NULL, &_err));
    offsetD = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       sizeof(cl_long) * num_kqueries, NULL, &_err));
    keysD = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                     sizeof(int) * num_kqueries, NULL, &_err));
    ansD = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(record) * num_kqueries, NULL, &_err));

    CL_CHECK(clEnqueueWriteBuffer(commandQueue, currKnodeD, CL_TRUE, 0,
                                  sizeof(cl_long) * num_kqueries, currKnode.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, offsetD, CL_TRUE, 0,
                                  sizeof(cl_long) * num_kqueries, offset.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, keysD, CL_TRUE, 0,
                                  sizeof(int) * num_kqueries, qkeys.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, ansD, CL_TRUE, 0,
                                  sizeof(record) * num_kqueries, ans.data(), 0, NULL, NULL));

    CL_CHECK(clSetKernelArg(findK_kernel, 0, sizeof(cl_long), &cl_height));
    CL_CHECK(clSetKernelArg(findK_kernel, 1, sizeof(cl_mem), &knodesD));
    CL_CHECK(clSetKernelArg(findK_kernel, 2, sizeof(cl_long), &cl_knodes_elem));
    CL_CHECK(clSetKernelArg(findK_kernel, 3, sizeof(cl_mem), &recordsD));
    CL_CHECK(clSetKernelArg(findK_kernel, 4, sizeof(cl_mem), &currKnodeD));
    CL_CHECK(clSetKernelArg(findK_kernel, 5, sizeof(cl_mem), &offsetD));
    CL_CHECK(clSetKernelArg(findK_kernel, 6, sizeof(cl_mem), &keysD));
    CL_CHECK(clSetKernelArg(findK_kernel, 7, sizeof(cl_mem), &ansD));

    size_t local_work_size = ORDER;
    size_t global_work_size = (size_t)num_kqueries * ORDER;
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, findK_kernel, 1, NULL,
                                    &global_work_size, &local_work_size, 0, NULL, NULL));
    CL_CHECK(clFinish(commandQueue));

    CL_CHECK(clEnqueueReadBuffer(commandQueue, ansD, CL_TRUE, 0,
                                 sizeof(record) * num_kqueries, ans.data(), 0, NULL, NULL));

    for (int i = 0; i < num_kqueries; ++i) {
      int expected = cpu_findK(K, recValues, qkeys[i]);
      if (ans[i].value != expected) {
        if (errors < 20)
          printf("*** findK error: query[%d] key=%d expected=%d actual=%d\n",
                 i, qkeys[i], expected, ans[i].value);
        ++errors;
      }
    }
  }

  //========================================================================
  //  findRangeK
  //========================================================================
  if (num_rqueries > 0) {
    std::vector<cl_long> currKnode(num_rqueries, 0);
    std::vector<cl_long> offset(num_rqueries, 0);
    std::vector<cl_long> lastKnode(num_rqueries, 0);
    std::vector<cl_long> offset2(num_rqueries, 0);
    std::vector<int> recstart(num_rqueries, 0);
    std::vector<int> reclen(num_rqueries, 0);

    currKnode2D = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                           sizeof(cl_long) * num_rqueries, NULL, &_err));
    offset2D = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        sizeof(cl_long) * num_rqueries, NULL, &_err));
    lastKnodeD = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          sizeof(cl_long) * num_rqueries, NULL, &_err));
    offset22D = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                         sizeof(cl_long) * num_rqueries, NULL, &_err));
    startD = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      sizeof(int) * num_rqueries, NULL, &_err));
    endD = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                    sizeof(int) * num_rqueries, NULL, &_err));
    recstartD = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                         sizeof(int) * num_rqueries, NULL, &_err));
    reclenD = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       sizeof(int) * num_rqueries, NULL, &_err));

    CL_CHECK(clEnqueueWriteBuffer(commandQueue, currKnode2D, CL_TRUE, 0,
                                  sizeof(cl_long) * num_rqueries, currKnode.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, offset2D, CL_TRUE, 0,
                                  sizeof(cl_long) * num_rqueries, offset.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, lastKnodeD, CL_TRUE, 0,
                                  sizeof(cl_long) * num_rqueries, lastKnode.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, offset22D, CL_TRUE, 0,
                                  sizeof(cl_long) * num_rqueries, offset2.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, startD, CL_TRUE, 0,
                                  sizeof(int) * num_rqueries, rstart.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, endD, CL_TRUE, 0,
                                  sizeof(int) * num_rqueries, rend.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, recstartD, CL_TRUE, 0,
                                  sizeof(int) * num_rqueries, recstart.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, reclenD, CL_TRUE, 0,
                                  sizeof(int) * num_rqueries, reclen.data(), 0, NULL, NULL));

    CL_CHECK(clSetKernelArg(findRangeK_kernel, 0, sizeof(cl_long), &cl_height));
    CL_CHECK(clSetKernelArg(findRangeK_kernel, 1, sizeof(cl_mem), &knodesD));
    CL_CHECK(clSetKernelArg(findRangeK_kernel, 2, sizeof(cl_long), &cl_knodes_elem));
    CL_CHECK(clSetKernelArg(findRangeK_kernel, 3, sizeof(cl_mem), &currKnode2D));
    CL_CHECK(clSetKernelArg(findRangeK_kernel, 4, sizeof(cl_mem), &offset2D));
    CL_CHECK(clSetKernelArg(findRangeK_kernel, 5, sizeof(cl_mem), &lastKnodeD));
    CL_CHECK(clSetKernelArg(findRangeK_kernel, 6, sizeof(cl_mem), &offset22D));
    CL_CHECK(clSetKernelArg(findRangeK_kernel, 7, sizeof(cl_mem), &startD));
    CL_CHECK(clSetKernelArg(findRangeK_kernel, 8, sizeof(cl_mem), &endD));
    CL_CHECK(clSetKernelArg(findRangeK_kernel, 9, sizeof(cl_mem), &recstartD));
    CL_CHECK(clSetKernelArg(findRangeK_kernel, 10, sizeof(cl_mem), &reclenD));

    size_t local_work_size = ORDER;
    size_t global_work_size = (size_t)num_rqueries * ORDER;
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, findRangeK_kernel, 1, NULL,
                                    &global_work_size, &local_work_size, 0, NULL, NULL));
    CL_CHECK(clFinish(commandQueue));

    CL_CHECK(clEnqueueReadBuffer(commandQueue, recstartD, CL_TRUE, 0,
                                 sizeof(int) * num_rqueries, recstart.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(commandQueue, reclenD, CL_TRUE, 0,
                                 sizeof(int) * num_rqueries, reclen.data(), 0, NULL, NULL));

    for (int i = 0; i < num_rqueries; ++i) {
      // Record indices are the sorted ranks; length spans [start, end].
      int exp_start = rstart_rank[i];
      int exp_len = rend_rank[i] - rstart_rank[i] + 1;
      if (recstart[i] != exp_start || reclen[i] != exp_len) {
        if (errors < 20)
          printf("*** findRangeK error: query[%d] [%d,%d] expected(start=%d,len=%d) actual(start=%d,len=%d)\n",
                 i, rstart[i], rend[i], exp_start, exp_len, recstart[i], reclen[i]);
        ++errors;
      }
    }
  }

  cleanup();
  if (errors != 0) {
    printf("FAILED! - %d errors\n", errors);
    return errors;
  }
  printf("PASSED!\n");
  return 0;
}
