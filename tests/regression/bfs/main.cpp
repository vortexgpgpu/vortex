#include <iostream>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <vortex2.h>
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
vx_buffer_h nextmask_buffer = nullptr;
vx_buffer_h visit_buffer = nullptr;
vx_buffer_h frontier_buffer = nullptr;
vx_buffer_h cost_buffer = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
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
    if (nodes_buffer)    vx_buffer_release(nodes_buffer);
    if (edges_buffer)    vx_buffer_release(edges_buffer);
    if (nextmask_buffer) vx_buffer_release(nextmask_buffer);
    if (visit_buffer)    vx_buffer_release(visit_buffer);
    if (frontier_buffer) vx_buffer_release(frontier_buffer);
    if (cost_buffer)     vx_buffer_release(cost_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
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
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  // Generate graph
  int max_edges_per_node = 5;
  std::vector<Node> node_data;
  std::vector<int> edge_data;
  generate_random_graph(size, max_edges_per_node, node_data, edge_data);

  // Assignments
  uint32_t num_nodes = size;
  uint32_t num_edges = edge_data.size();

  uint32_t nodes_buf_size    = num_nodes * sizeof(Node);
  uint32_t edges_buf_size    = num_edges * sizeof(int32_t);
  uint32_t nextmask_buf_size = num_nodes * sizeof(uint8_t);
  uint32_t visit_buf_size    = num_nodes * sizeof(uint8_t);
  uint32_t frontier_buf_size = num_nodes * sizeof(uint32_t);
  uint32_t cost_buf_size     = num_nodes * sizeof(int32_t);

  std::cout << "number of nodes: " << num_nodes << std::endl;

  kernel_arg.num_nodes = num_nodes;
  kernel_arg.num_edges = num_edges;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_buffer_create(device, nodes_buf_size,    VX_MEM_READ_WRITE, &nodes_buffer));
  RT_CHECK(vx_buffer_address(nodes_buffer,    &kernel_arg.nodes_addr));
  RT_CHECK(vx_buffer_create(device, edges_buf_size,    VX_MEM_READ_WRITE, &edges_buffer));
  RT_CHECK(vx_buffer_address(edges_buffer,    &kernel_arg.edges_addr));
  RT_CHECK(vx_buffer_create(device, nextmask_buf_size, VX_MEM_READ_WRITE, &nextmask_buffer));
  RT_CHECK(vx_buffer_address(nextmask_buffer, &kernel_arg.nextmask_addr));
  RT_CHECK(vx_buffer_create(device, visit_buf_size,    VX_MEM_READ_WRITE, &visit_buffer));
  RT_CHECK(vx_buffer_address(visit_buffer,    &kernel_arg.visit_addr));
  RT_CHECK(vx_buffer_create(device, frontier_buf_size, VX_MEM_READ_WRITE, &frontier_buffer));
  RT_CHECK(vx_buffer_address(frontier_buffer, &kernel_arg.frontier_addr));
  RT_CHECK(vx_buffer_create(device, cost_buf_size,     VX_MEM_READ_WRITE, &cost_buffer));
  RT_CHECK(vx_buffer_address(cost_buffer,     &kernel_arg.cost_addr));

  // host buffers
  std::vector<Node>     h_nodes(num_nodes);
  std::vector<int32_t>  h_edges(num_edges);
  std::vector<uint8_t>  h_nextmask(num_nodes, 0);
  std::vector<uint8_t>  h_visit(num_nodes, 0);
  std::vector<uint32_t> h_frontier;
  std::vector<int32_t>  h_cost(num_nodes, -1);

  for (uint32_t i = 0; i < num_nodes; i++) h_nodes[i] = node_data[i];
  for (uint32_t i = 0; i < num_edges; i++) h_edges[i] = edge_data[i];

  // source node = 0
  h_visit[0] = 1;
  h_cost[0]  = 0;
  h_frontier.push_back(0);

  // upload static data once
  RT_CHECK(vx_enqueue_write(queue, nodes_buffer, 0, h_nodes.data(), nodes_buf_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, edges_buffer, 0, h_edges.data(), edges_buf_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, cost_buffer,  0, h_cost.data(),  cost_buf_size, 0, nullptr, nullptr));

  // load kernel module
  std::cout << "load kernel module" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  // BFS level-by-level dispatch
  std::cout << "start device" << std::endl;
  while (!h_frontier.empty()) {
    uint32_t frontier_size = (uint32_t)h_frontier.size();

    kernel_arg.frontier_size = frontier_size;

    RT_CHECK(vx_enqueue_write(queue, frontier_buffer, 0, h_frontier.data(), frontier_size * sizeof(uint32_t), 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_write(queue, visit_buffer,    0, h_visit.data(),    visit_buf_size, 0, nullptr, nullptr));
    std::fill(h_nextmask.begin(), h_nextmask.end(), 0);
    RT_CHECK(vx_enqueue_write(queue, nextmask_buffer, 0, h_nextmask.data(), nextmask_buf_size, 0, nullptr, nullptr));

    vx_event_h launch_ev = nullptr, read_ev0 = nullptr, read_ev1 = nullptr;
    {
      uint32_t grid_dim[1], block_dim[1];
      RT_CHECK(vx_device_max_occupancy_grid(device, 1, &frontier_size, grid_dim, block_dim));
      vx_launch_info_t li = {};
      li.struct_size  = sizeof(li);
      li.kernel       = kernel;
      li.args_host    = &kernel_arg;
      li.args_size    = sizeof(kernel_arg);
      li.ndim         = 1;
      li.grid_dim[0]  = grid_dim[0];
      li.block_dim[0] = block_dim[0];
      RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
    }

    // compact next frontier on host
    RT_CHECK(vx_enqueue_read(queue, h_nextmask.data(), nextmask_buffer, 0, nextmask_buf_size, 1, &launch_ev, &read_ev0));
    RT_CHECK(vx_enqueue_read(queue, h_cost.data(),     cost_buffer,     0, cost_buf_size, 1, &launch_ev, &read_ev1));
    RT_CHECK(vx_event_wait_value(read_ev1, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev1);
    vx_event_release(read_ev0);
    vx_event_release(launch_ev);
    h_frontier.clear();
    for (uint32_t v = 0; v < num_nodes; ++v) {
      if (h_nextmask[v] && !h_visit[v]) {
        h_visit[v] = 1;
        h_frontier.push_back(v);
      }
    }
  }

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
