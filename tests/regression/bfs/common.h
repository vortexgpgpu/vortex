#ifndef _COMMON_H_
#define _COMMON_H_

#define MAX_NODES 100000

struct Node {
  int starting;
  int no_of_edges;
};

typedef struct {
  uint32_t num_nodes;
  uint32_t num_edges;

  uint64_t nodes_addr;  // Node          Buffer
  uint64_t edges_addr;  // Edge          Buffer
  uint64_t mask_addr;  // Mask          Buffer
  uint64_t nextmask_addr;  // Update Mask   Buffer
  uint64_t visit_addr;  // Visited       Buffer
  uint64_t frontier_addr;  // Thread Update Buffer
  uint64_t cost_addr;   // Cost Buffer
} kernel_arg_t;

#endif
