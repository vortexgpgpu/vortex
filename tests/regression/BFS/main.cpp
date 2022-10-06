#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vortex.h>
#include <vector>
#include "common.h"

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
	 cleanup();			                                              \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

const char* kernel_file = "kernel.bin";
uint32_t count = 0;

Node* h_graph_nodes;
int* h_graph_edges;
char* h_graph_mask;
char* h_updating_graph_mask;
char* h_graph_visited;
int* h_cost;
int no_of_nodes;

vx_device_h device = nullptr;
vx_buffer_h arg_buf = nullptr;
vx_buffer_h common_buf = nullptr;

kernel_arg_t kernel_arg;

void cleanup() {
  if (arg_buf) {
    vx_buf_free(arg_buf);
  }
  if (device) {
    vx_mem_free(device, kernel_arg.graphnodes_addr);
    vx_mem_free(device, kernel_arg.graphmask_addr);
    vx_mem_free(device, kernel_arg.graphupmask_addr);
    vx_mem_free(device, kernel_arg.graphvisited_addr);
    vx_mem_free(device, kernel_arg.graphedges_addr);
    vx_mem_free(device, kernel_arg.gcost_addr);
    vx_mem_free(device, kernel_arg.hover_addr);
    vx_dev_close(device);
  }
}

int main(void) { 
  size_t value;
  //0. Read in Graph from a file
    int no_of_nodes;
    FILE *fp;
	char *input_f = "graph4.txt";
	fp = fopen(input_f, "r");
	if (!fp) {
	return 0;
	}
	int source = 0;
	fscanf(fp,"%d",&no_of_nodes);
	std::cout <<"Reading File completed!\n"<< std::endl;
//1. Allocate and initialize host memory
	int edge_list_size;
	Node *h_graph_nodes;
	int *h_graph_mask, *h_updating_graph_mask, *h_graph_visited;
	h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	h_graph_mask = (int*) malloc(sizeof(int)*no_of_nodes);
	h_updating_graph_mask = (int*) malloc(sizeof(int)*no_of_nodes);
	h_graph_visited = (int*) malloc(sizeof(int)*no_of_nodes);
        int start, edgeno;
        // initalize the memory
        for(int i = 0; i < no_of_nodes; i++) {
            fscanf(fp,"%d %d",&start,&edgeno);
            h_graph_nodes[i].starting = start;
            h_graph_nodes[i].no_of_edges = edgeno;
            h_graph_mask[i]=0;
            h_updating_graph_mask[i]=0;
            h_graph_visited[i]=0;
        }
        //read the source node from the file
        fscanf(fp,"%d",&source);
        source=0;
        //set the source node as true in the mask
        h_graph_mask[source]=1;
        h_graph_visited[source]=1;
        fscanf(fp,"%d",&edge_list_size);
        int id,cost;
        int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
        for(int i=0; i < edge_list_size ; i++) {
            fscanf(fp,"%d",&id);
            fscanf(fp,"%d",&cost);
            h_graph_edges[i] = id;
        }
        if(fp)
            fclose(fp);
    // Allocate mem for the result on host side
		int	*h_cost = (int*) malloc(sizeof(int)*no_of_nodes);
        int *h_cost_ref = (int*)malloc(sizeof(int)*no_of_nodes);
        for(int i=0; i<no_of_nodes; i++) {
            h_cost[i] = -1;
            h_cost_ref[i] = -1;
        }
        h_cost[source]=0;
        h_cost_ref[source]=0;
        
  //2. open device connection
  std::cout << "open device connection" << std::endl;  
  RT_CHECK(vx_dev_open(&device));

  //3. Declare buffer sizes
  uint32_t num_points = no_of_nodes;
  uint32_t graphnodes_bufsz = sizeof(Node)*no_of_nodes;
  uint32_t common_bufsz =  sizeof(int)*no_of_nodes; 
  uint32_t hover_bufsz =  sizeof(int); 
  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "buffer size: " << common_bufsz << " bytes" << std::endl;

  //4. upload program
  std::cout << "upload program" << std::endl;  
  RT_CHECK(vx_upload_kernel_file(device, kernel_file));

  //5. allocate device memory
  std::cout << "allocate device memory" << std::endl;  

  RT_CHECK(vx_mem_alloc(device, graphnodes_bufsz, &value));
  kernel_arg.graphnodes_addr = value;
  
  RT_CHECK(vx_mem_alloc(device, common_bufsz, &value));
  kernel_arg.graphmask_addr = value;

  RT_CHECK(vx_mem_alloc(device, common_bufsz, &value));
  kernel_arg.graphupmask_addr = value;
  
  RT_CHECK(vx_mem_alloc(device, common_bufsz, &value));
  kernel_arg.graphvisited_addr = value;

  RT_CHECK(vx_mem_alloc(device, common_bufsz, &value));
  kernel_arg.graphedges_addr = value;

  RT_CHECK(vx_mem_alloc(device, common_bufsz, &value));
  kernel_arg.gcost_addr = value;

  RT_CHECK(vx_mem_alloc(device, hover_bufsz, &value));
  kernel_arg.hover_addr = value;
  
  kernel_arg.no_of_nodes = no_of_nodes; 
  
  std::cout << "cost_dst=" << std::hex << kernel_arg.gcost_addr << std::endl;
  
int iter = 0;
int h_over = 0; //0=False, 1=True
  //6. allocate shared memory  
  std::cout << "allocate shared memory" << std::endl;
  uint32_t shared_bufsz = std::max<uint32_t>(graphnodes_bufsz, common_bufsz);
  RT_CHECK(vx_buf_alloc(device, sizeof(kernel_arg_t), &arg_buf));
  RT_CHECK(vx_buf_alloc(device, shared_bufsz, &common_buf));

do {
	h_over = 0;

//I. 7. upload kernel 1 arguments 
	kernel_arg.testid = 0;
	{
	auto buf_ptr_upload = (int*)vx_host_ptr(common_buf);
      buf_ptr_upload[0] = h_over;
  }
   RT_CHECK(vx_copy_to_dev(common_buf, kernel_arg.hover_addr, hover_bufsz, 0));  
  
  std::cout << "upload kernel argument" << std::endl;
  {
    auto buf_ptr = (int*)vx_host_ptr(arg_buf);
    memcpy(buf_ptr, &kernel_arg, sizeof(kernel_arg_t));
    RT_CHECK(vx_copy_to_dev(arg_buf, KERNEL_ARG_DEV_MEM_ADDR, sizeof(kernel_arg_t), 0));
  }

//I. 8. upload source buffers
{
	auto buf_ptr_upload = (Node*)vx_host_ptr(common_buf);
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr_upload[i] = h_graph_nodes[i];
    }
}
RT_CHECK(vx_copy_to_dev(common_buf, kernel_arg.graphnodes_addr, graphnodes_bufsz, 0));

{
	auto buf_ptr_upload = (int32_t*)vx_host_ptr(common_buf);
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr_upload[i] = h_graph_mask[i];
    }
}
	RT_CHECK(vx_copy_to_dev(common_buf, kernel_arg.graphmask_addr, common_bufsz, 0));

{
	auto buf_ptr_upload = (int32_t*)vx_host_ptr(common_buf);
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr_upload[i] = h_updating_graph_mask[i];
    }
}
	RT_CHECK(vx_copy_to_dev(common_buf, kernel_arg.graphupmask_addr, common_bufsz, 0));
	
{
	auto buf_ptr_upload = (int32_t*)vx_host_ptr(common_buf);
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr_upload[i] = h_graph_visited[i];
    }
}
	RT_CHECK(vx_copy_to_dev(common_buf, kernel_arg.graphvisited_addr, common_bufsz, 0));

{
	auto buf_ptr_upload = (int32_t*)vx_host_ptr(common_buf);
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr_upload[i] = h_graph_edges[i];
    }
}
	RT_CHECK(vx_copy_to_dev(common_buf, kernel_arg.graphedges_addr,common_bufsz, 0));

//I. 9. Initialize destination buffer
{
	auto buf_ptr_cost = (int32_t*)vx_host_ptr(common_buf);
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr_cost[i] = h_cost[i];
    }
}
RT_CHECK(vx_copy_to_dev(common_buf, kernel_arg.gcost_addr, common_bufsz, 0));
for (uint32_t i = 0; i < no_of_nodes; ++i) {
      std::cout << "init cost at index " <<i<<" is "<<h_cost[i]<<" on iter "<<iter<<std::endl;
    }

//I. 10. Start device
	std::cout << "start device" << std::endl;
	RT_CHECK(vx_start(device));

//I. 11. Wait for completion
    std::cout << "wait for completion" << std::endl;
    RT_CHECK(vx_ready_wait(device, MAX_TIMEOUT));
//I. 12. Download destination buffer for masks
//I. 13. Copy results to host arrays
    std::cout << "download destination buffer" << std::endl;
    RT_CHECK(vx_copy_from_dev(common_buf, kernel_arg.graphmask_addr, common_bufsz, 0));
    {
    auto buf_ptr1 = (int32_t*)vx_host_ptr(common_buf);
    for (uint32_t i = 0; i < no_of_nodes; ++i) {   
        h_graph_mask[i] = buf_ptr1[i];
    }
}
    RT_CHECK(vx_copy_from_dev(common_buf, kernel_arg.graphupmask_addr, common_bufsz, 0));
    {
auto buf_ptr1 = (int32_t*)vx_host_ptr(common_buf);
for (uint32_t i = 0; i < no_of_nodes; ++i) {   
      h_updating_graph_mask[i] = buf_ptr1[i];
    }
}
    RT_CHECK(vx_copy_from_dev(common_buf, kernel_arg.graphvisited_addr, common_bufsz, 0));
    {
auto buf_ptr1 = (int32_t*)vx_host_ptr(common_buf);
for (uint32_t i = 0; i < no_of_nodes; ++i) {   
      h_graph_visited[i] = buf_ptr1[i];
    }
}

//I. 12. download destination buffer - cost
//    std::cout << "download destination buffer" << std::endl;
RT_CHECK(vx_copy_from_dev(common_buf, kernel_arg.gcost_addr, common_bufsz, 0));
//I. 13. Copy results
{
auto buf_ptr = (int32_t*)vx_host_ptr(common_buf);
for (uint32_t i = 0; i < no_of_nodes; ++i) {
      h_cost[i] =  buf_ptr[i];
      std::cout << "cost at index " <<i<<" is "<<h_cost[i]<<" on iter "<<iter<<std::endl;
    }
}
/*************** KERNEL 2 *************/
   
  //II. 7. upload kernel argument
  kernel_arg.testid = 1;
  std::cout << "upload kernel argument" << std::endl;
  {
    auto buf_ptr = (int*)vx_host_ptr(arg_buf);
    memcpy(buf_ptr, &kernel_arg, sizeof(kernel_arg_t));
    RT_CHECK(vx_copy_to_dev(arg_buf, KERNEL_ARG_DEV_MEM_ADDR, sizeof(kernel_arg_t), 0));
  }
// Upload h_over
{
	auto buf_ptr_upload = (int*)vx_host_ptr(common_buf);
      buf_ptr_upload[0] = h_over;
}
RT_CHECK(vx_copy_to_dev(common_buf, kernel_arg.hover_addr, hover_bufsz, 0)); 
//II. 8. upload source buffers
{
	auto buf_ptr_upload = (int32_t*)vx_host_ptr(common_buf);
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr_upload[i] = h_graph_mask[i];
    }
}
	RT_CHECK(vx_copy_to_dev(common_buf, kernel_arg.graphmask_addr, common_bufsz, 0));

{
	auto buf_ptr_upload = (int32_t*)vx_host_ptr(common_buf);
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr_upload[i] = h_updating_graph_mask[i];
    }
}
	RT_CHECK(vx_copy_to_dev(common_buf, kernel_arg.graphupmask_addr, common_bufsz, 0));
	
{
	auto buf_ptr_upload = (int32_t*)vx_host_ptr(common_buf);
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr_upload[i] = h_graph_visited[i];
    }
}
	RT_CHECK(vx_copy_to_dev(common_buf, kernel_arg.graphvisited_addr, common_bufsz, 0));

//II. 10. Start device
	std::cout << "start device" << std::endl;
	RT_CHECK(vx_start(device));

//II. 11. Wait for completion
    std::cout << "wait for completion" << std::endl;
    RT_CHECK(vx_ready_wait(device, MAX_TIMEOUT));
//II. 12. Download destination buffer for masks
    std::cout << "download destination buffer" << std::endl;
    RT_CHECK(vx_copy_from_dev(common_buf, kernel_arg.graphmask_addr, common_bufsz, 0));
    {
    auto buf_ptr1 = (int32_t*)vx_host_ptr(common_buf);
    for (uint32_t i = 0; i < no_of_nodes; ++i) {   
        h_graph_mask[i] = buf_ptr1[i];
    }
}
    RT_CHECK(vx_copy_from_dev(common_buf, kernel_arg.graphupmask_addr, common_bufsz, 0));
    {
auto buf_ptr1 = (int32_t*)vx_host_ptr(common_buf);
for (uint32_t i = 0; i < no_of_nodes; ++i) {   
      h_updating_graph_mask[i] = buf_ptr1[i];
std::cout << "hup mask II at index " <<i<<" is "<<h_updating_graph_mask[i]<<" on iter "<<iter<<std::endl;
    }
}
    RT_CHECK(vx_copy_from_dev(common_buf, kernel_arg.graphvisited_addr, common_bufsz, 0));
    {
auto buf_ptr1 = (int32_t*)vx_host_ptr(common_buf);
for (uint32_t i = 0; i < no_of_nodes; ++i) {   
      h_graph_visited[i] = buf_ptr1[i];
    }
}

/*************** KERNEL 2 *************/
RT_CHECK(vx_copy_from_dev(common_buf, kernel_arg.hover_addr, hover_bufsz, 0));
   {
    auto buf_ptr = (int32_t*)vx_host_ptr(common_buf);     
    h_over = buf_ptr[0]; 
    std::cout << "HOVER IS " <<h_over<<std::endl;
    }
iter++;
}while(h_over);

//12. download destination buffer
    std::cout << "download destination buffer" << std::endl;
    RT_CHECK(vx_copy_from_dev(common_buf, kernel_arg.gcost_addr, common_bufsz, 0));

//13. Printing results
auto buf_ptr = (int32_t*)vx_host_ptr(common_buf);
for (uint32_t i = 0; i < 4; i++) {
      int cur = buf_ptr[i];
      std::cout << "array index [" <<i<<"]is "<<cur<<std::endl;
    }

  return 0;
}
