#include <iostream>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <vortex.h>
#include "common.h"

#define FLOAT_ULP 6

#define RT_CHECK(_expr)                                      \
  do {                                                       \
    int _ret = _expr;                                        \
    if (0 == _ret)                                           \
      break;                                                 \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
    cleanup();                                               \
    exit(-1);                                                \
  } while (false)

///////////////////////////////////////////////////////////////////////////////

template <typename Type>
class Comparator {};

template <>
class Comparator<int> {
public:
  static const char *type_str() {
    return "integer";
  }
  static int generate() {
    return rand() % 100;
  }
  static bool compare(int a, int b, int index, int errors) {
    if (a != b) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%d, actual=%d\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<float> {
private:
  union Float_t {
    float f;
    int i;
  };

public:
  static const char *type_str() {
    return "float";
  }
  static float generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {
    union fi_t {
      float f;
      int32_t i;
    };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

const char *kernel_file = "kernel.vxbin";
uint32_t size = 1024;

vx_device_h device = nullptr;
vx_buffer_h nodes_buffer = nullptr;
vx_buffer_h edges_buffer = nullptr;
vx_buffer_h mask_buffer = nullptr;
vx_buffer_h nextmask_buffer = nullptr;
vx_buffer_h visit_buffer = nullptr;
vx_buffer_h frontier_buffer = nullptr;
vx_buffer_h cost_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Vortex Test." << std::endl;
  std::cout << "Usage: [-k: kernel] [-n words] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'h':
      show_usage();
      exit(0);
      break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(nodes_buffer);
    vx_mem_free(edges_buffer);
    vx_mem_free(mask_buffer);
    vx_mem_free(nextmask_buffer);
    vx_mem_free(visit_buffer);
    vx_mem_free(frontier_buffer);
    vx_mem_free(cost_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

void generate_random_graph(int num_nodes, int max_edges_per_node,
                           std::vector<Node> &nodes, std::vector<int> &edges) {
  nodes.resize(num_nodes);
  edges.clear();

  int edge_count = 0;
  for (int i = 0; i < num_nodes; ++i) {
    nodes[i].starting = edge_count;
    int num_edges = rand() % (max_edges_per_node + 1);
    nodes[i].no_of_edges = num_edges;

    for (int e = 0; e < num_edges; ++e) {
      int dest_node = Comparator<int32_t>::generate() % num_nodes;
      edges.push_back(dest_node);
      edge_count++;
    }
  }
}

static void bfs_cpu(Node graph_nodes[], int graph_edges[], int num_nodes, int source, int cost[]) {
  int queue[MAX_NODES];
  int head = 0, tail = 0;

  // Initialize cost array to -1 (unvisited)
  for (int i = 0; i < num_nodes; i++) {
    cost[i] = -1;
  }

  // Start node cost = 0 and enqueue it
  cost[source] = 0;
  queue[tail++] = source;

  while (head < tail) {
    int current = queue[head++];
    Node node = graph_nodes[current];
    int start = node.starting;
    int end = start + node.no_of_edges;

    for (int i = start; i < end; i++) {
      int neighbor = graph_edges[i];
      if (cost[neighbor] == -1) { // Not visited yet
        cost[neighbor] = cost[current] + 1;
        queue[tail++] = neighbor;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  std::srand(50);

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  // Generate graph
  int max_edges_per_node = 5;
  std::vector<Node> node_data;
  std::vector<int> edge_data;
  generate_random_graph(size, max_edges_per_node, node_data, edge_data);

  // Assignments
  uint32_t num_nodes = size;
  uint32_t num_edges = edge_data.size();

  uint32_t nodes_buf_size = num_nodes * sizeof(Node);
  uint32_t edges_buf_size = num_edges * sizeof(int32_t);
  uint32_t mask_buf_size      = num_nodes * sizeof(uint8_t);
  uint32_t nextmask_buf_size = num_nodes * sizeof(uint8_t);
  uint32_t visit_buf_size     = num_nodes * sizeof(uint8_t);
  uint32_t frontier_buf_size = num_nodes * sizeof(int32_t);
  uint32_t cost_buf_size = num_nodes * sizeof(int32_t);

  std::cout << "number of nodes: " << num_nodes << std::endl;

  kernel_arg.num_nodes = num_nodes;
  kernel_arg.num_edges = num_edges;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, nodes_buf_size, VX_MEM_READ_WRITE, &nodes_buffer));
  RT_CHECK(vx_mem_address(nodes_buffer, &kernel_arg.nodes_addr));
  RT_CHECK(vx_mem_alloc(device, edges_buf_size, VX_MEM_READ_WRITE, &edges_buffer));
  RT_CHECK(vx_mem_address(edges_buffer, &kernel_arg.edges_addr));
  RT_CHECK(vx_mem_alloc(device, mask_buf_size, VX_MEM_READ_WRITE, &mask_buffer));
  RT_CHECK(vx_mem_address(mask_buffer, &kernel_arg.mask_addr));
  RT_CHECK(vx_mem_alloc(device, nextmask_buf_size, VX_MEM_READ_WRITE, &nextmask_buffer));
  RT_CHECK(vx_mem_address(nextmask_buffer, &kernel_arg.nextmask_addr));
  RT_CHECK(vx_mem_alloc(device, visit_buf_size, VX_MEM_READ_WRITE, &visit_buffer));
  RT_CHECK(vx_mem_address(visit_buffer, &kernel_arg.visit_addr));
  RT_CHECK(vx_mem_alloc(device, frontier_buf_size, VX_MEM_READ_WRITE, &frontier_buffer));
  RT_CHECK(vx_mem_address(frontier_buffer, &kernel_arg.frontier_addr));

  RT_CHECK(vx_mem_alloc(device, cost_buf_size, VX_MEM_READ_WRITE, &cost_buffer));
  RT_CHECK(vx_mem_address(cost_buffer, &kernel_arg.cost_addr));

  std::cout << "nodes_addr=0x" << std::hex << kernel_arg.nodes_addr << std::endl;
  std::cout << "edges_addr=0x" << std::hex << kernel_arg.edges_addr << std::endl;
  std::cout << "mask_addr=0x" << std::hex << kernel_arg.mask_addr << std::endl;
  std::cout << "nextmask_addr=0x" << std::hex << kernel_arg.nextmask_addr << std::endl;
  std::cout << "visit_addr=0x" << std::hex << kernel_arg.visit_addr << std::endl;
  std::cout << "frontier_addr=0x" << std::hex << kernel_arg.frontier_addr << std::endl;
  std::cout << "cost_addr=0x" << std::hex << kernel_arg.cost_addr << std::endl;

  // allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<Node>    h_nodes(num_nodes);
  std::vector<int32_t> h_edges(num_edges);
  std::vector<uint8_t> h_mask(num_nodes);
  std::vector<uint8_t> h_nextmask(num_nodes);
  std::vector<uint8_t> h_visit(num_nodes);
  std::vector<int32_t> h_frontier(num_nodes);
  std::vector<int32_t> h_cost(num_nodes);

  // Node
  for (uint32_t i = 0; i < num_nodes; i++) {
    h_nodes[i] = node_data[i];
  }

  // Edge
  for (uint32_t i = 0; i < num_edges; i++) {
    h_edges[i] = edge_data[i];
  }

  // Masks
  for (uint32_t i = 0; i < num_nodes; i++) {
    h_mask[i] = 0;
  }
  h_mask[0] = 1;

  // Visited
  for (uint32_t i = 0; i < num_nodes; i++) {
    h_visit[i] = 0;
  }
  h_visit[0] = 1;

  // Thread updated
  for (uint32_t i = 0; i < num_nodes; i++) {
    h_frontier[i] = 0;
  }

  // Cost
  for (uint32_t i = 0; i < num_nodes; i++) {
    h_cost[i] = -1;
  }
  h_cost[0] = 0;

  // upload source buffer0
  std::cout << "upload source buffer0" << std::endl;
  RT_CHECK(vx_copy_to_dev(nodes_buffer, h_nodes.data(), 0, nodes_buf_size));

  // upload source buffer1
  std::cout << "upload source buffer1" << std::endl;
  RT_CHECK(vx_copy_to_dev(edges_buffer, h_edges.data(), 0, edges_buf_size));

  // upload source buffer2
  std::cout << "upload source buffer2" << std::endl;
  RT_CHECK(vx_copy_to_dev(mask_buffer, h_mask.data(), 0, mask_buf_size));

  // upload source buffer3
  std::cout << "upload source buffer3" << std::endl;
  RT_CHECK(vx_copy_to_dev(nextmask_buffer, h_nextmask.data(), 0, nextmask_buf_size));

  // upload source buffer4
  std::cout << "upload source buffer4" << std::endl;
  RT_CHECK(vx_copy_to_dev(visit_buffer, h_visit.data(), 0, visit_buf_size));

  // upload source buffer5
  std::cout << "upload source buffer5" << std::endl;
  RT_CHECK(vx_copy_to_dev(frontier_buffer, h_frontier.data(), 0, frontier_buf_size));

  // upload cost/destination buffer
  std::cout << "upload destination buffer" << std::endl;
  RT_CHECK(vx_copy_to_dev(cost_buffer, h_cost.data(), 0, cost_buf_size));

  // upload program
  std::cout << "upload program" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_cost.data(), cost_buffer, 0, cost_buf_size));

  // verify result
  std::cout << "verify result" << std::endl;

  // Run Golden Results
  std::vector<int> cost(num_nodes);
  bfs_cpu(node_data.data(), edge_data.data(), num_nodes, 0, cost.data());

  // Check for errors
  int errors = 0;
  for (uint32_t i = 0; i < num_nodes; i++) {

    int cur = h_cost[i];
    int ref = cost[i];

    if (cur != ref) {
      std::cout << "error at result #" << std::dec << i
                << std::hex << ": actual=" << cur << ", expected=" << ref << std::endl;
      ++errors;
    }
  }

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}
