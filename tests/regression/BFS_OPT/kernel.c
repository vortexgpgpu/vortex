#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

typedef void (*PFN_Kernel)(int task_id, kernel_arg_t* arg);

void kernel_1(int __DIVERGENT__ task_id, kernel_arg_t* arg) {
        int tid = task_id;
        int index, id;
		//uint32_t count=0;
		//Get all needed args 
		Node* node_ptr = (Node*)arg->graphnodes_addr;
		int32_t* graphedges_ptr = (int32_t*)arg->graphedges_addr;
		int32_t* graphmask_ptr = (int32_t*)arg->graphmask_addr;
		int32_t* upgraphmask_ptr = (int32_t*)arg->graphupmask_addr;
		int32_t* graphvisited_ptr = (int32_t*)arg->graphvisited_addr;
		int32_t* cost_ptr = (int32_t*)arg->gcost_addr;
		int32_t* hover = (int32_t*)arg->hover_addr;
		int32_t* count = (int32_t*)arg->count_addr;
		int32_t* count1 = (int32_t*)arg->count1_addr;
		uint32_t num_points = arg->no_of_nodes;	
                int local_count = (int)count[0];
		//cost_ptr[0] = local_count;
		if(tid<count1[0])
		{
		index = graphmask_ptr[tid];
		for(int i=node_ptr[index].starting; i<(node_ptr[index].no_of_edges + node_ptr[index].starting); i++)
			{ 
				 id = graphedges_ptr[i]; //where is it connected to
				 if(!graphvisited_ptr[id]){
				        cost_ptr[id] = cost_ptr[index]+1; //add the cost
					upgraphmask_ptr[local_count]=id; //add the neighbour to active list
					local_count = local_count+1;
					//vx_barrier(0, arg->NW);
					//count[0] = count[0]+1;
					//count[0] = local_count+1;
					//cost_ptr[id] = local_count;
					hover[0]=1;
					graphvisited_ptr[id]=1;
			          }
		         }
	        }

count[0] = (int)local_count;
count1[0] = (int)local_count;
//cost_ptr[0] = count[0];
}

static const PFN_Kernel sc_tests[] = {
	kernel_1,
	//kernel_2,
};

void main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(arg->no_of_nodes, (vx_spawn_tasks_cb)sc_tests[arg->testid], arg);
}
