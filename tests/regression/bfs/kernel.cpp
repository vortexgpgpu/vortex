#include <vx_spawn.h>
#include "common.h"
#include "vx_print.h"

static void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
  auto *__restrict nodes = reinterpret_cast<Node *>(arg->nodes_addr);
  auto *__restrict edges = reinterpret_cast<int32_t *>(arg->edges_addr);
  auto *__restrict visit = reinterpret_cast<uint8_t *>(arg->visit_addr);
  auto *__restrict nextmask = reinterpret_cast<uint8_t *>(arg->nextmask_addr);
  auto *__restrict frontier = reinterpret_cast<uint32_t *>(arg->frontier_addr);
  auto *__restrict cost = reinterpret_cast<int32_t *>(arg->cost_addr);

  uint32_t tid = blockIdx.x;
  uint32_t v = frontier[tid];
  uint32_t start = nodes[v].starting;
  uint32_t deg = nodes[v].no_of_edges;
  uint32_t end = start + deg;

  int32_t cv = cost[v] + 1;

  for (uint32_t i = start; i < end; ++i) {
    uint32_t nid = edges[i];
    if (!visit[nid]) {    // skips already-visited from prior levels
      nextmask[nid] = 1;  // idempotent
      cost[nid] = cv;     // same cv from any parent at this level
    }
  }
}

inline uint32_t build_initial_frontier_from_mask(uint8_t *__restrict mask,
                                                 uint8_t *__restrict visit,
                                                 uint32_t *__restrict frontier,
                                                 uint32_t __UNIFORM__ num_nodes) {
  uint32_t out = 0;
  for (uint32_t v = 0; v < num_nodes; ++v) {
    if (!mask[v])
      continue;
    frontier[out++] = v;
    visit[v] = 1;
    mask[v] = 0;
  }
  return out;
}

inline uint32_t compact_next_frontier(uint8_t *__restrict next_mask,
                                      uint8_t *__restrict visit,
                                      uint32_t *__restrict frontier_out,
                                      uint32_t __UNIFORM__ num_nodes) {
  uint32_t out = 0;
  for (uint32_t v = 0; v < num_nodes; ++v) {
    if (next_mask[v] && !visit[v]) {
      visit[v] = 1;
      frontier_out[out++] = v;
    }
    next_mask[v] = 0;
  }
  return out;
}

int main() {
  auto *__UNIFORM__ arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);

  auto *__restrict mask = reinterpret_cast<uint8_t *>(arg->mask_addr);
  auto *__restrict next_mask = reinterpret_cast<uint8_t *>(arg->nextmask_addr);
  auto *__restrict visit = reinterpret_cast<uint8_t *>(arg->visit_addr);
  auto *__restrict frontier = reinterpret_cast<uint32_t *>(arg->frontier_addr);

  const uint32_t N = arg->num_nodes;

  for (uint32_t i = 0; i < N; ++i) {
    next_mask[i] = 0;
  }

  uint32_t frontier_size = build_initial_frontier_from_mask(mask, visit, frontier, N);
  while (frontier_size > 0) {
    uint32_t grid_dim[1] = {frontier_size};
    uint32_t block_dim[1] = {1};
    vx_spawn_threads(1, grid_dim, block_dim, (vx_kernel_func_cb)kernel_body, arg);
    frontier_size = compact_next_frontier(next_mask, visit, frontier, N);
  }
}
