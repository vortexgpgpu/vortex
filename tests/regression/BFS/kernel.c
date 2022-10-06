#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

typedef void (*PFN_Kernel)(int task_id, kernel_arg_t* arg);

void kernel_1(int __DIVERGENT__ task_id, kernel_arg_t* arg) {
        int tid = task_id;
	//int tid = vx_thread_id();
				//Get all needed args 
        			Node* node_ptr = (Node*)arg->graphnodes_addr;
				int32_t* graphedges_ptr = (int32_t*)arg->graphedges_addr;
				int32_t* graphmask_ptr = (int32_t*)arg->graphmask_addr;
				int32_t* upgraphmask_ptr = (int32_t*)arg->graphupmask_addr;
				int32_t* graphvisited_ptr = (int32_t*)arg->graphvisited_addr;
				int32_t* cost_ptr = (int32_t*)arg->gcost_addr;
        			uint32_t num_points = arg->no_of_nodes;		
if( tid<num_points && graphmask_ptr[tid])
{
		graphmask_ptr[tid]=0;
		
		for(int i=node_ptr[tid].starting; i<(node_ptr[tid].no_of_edges + node_ptr[tid].starting);i++)
		{

					int id = graphedges_ptr[i];
					if(!graphvisited_ptr[id])
				{
					        cost_ptr[id]=cost_ptr[tid]+1;
						upgraphmask_ptr[id]=1;
				}

		}
}
}


void kernel_2(int __DIVERGENT__ task_id, kernel_arg_t* arg) {
  int tid = task_id;
//int tid = vx_thread_id();
  //Get all needed args
  int32_t* graphmask_ptr = (int32_t*)arg->graphmask_addr;
	int32_t* upgraphmask_ptr = (int32_t*)arg->graphupmask_addr;
	int32_t* graphvisited_ptr = (int32_t*)arg->graphvisited_addr;
	int32_t* hover = (int32_t*)arg->hover_addr;
	uint32_t num_points = arg->no_of_nodes;
  if( tid<num_points && upgraphmask_ptr[tid]){
		graphmask_ptr[tid]=1;
		graphvisited_ptr[tid]=1;
		hover[0]=1;
		upgraphmask_ptr[tid]=0;
	}
}

static const PFN_Kernel sc_tests[] = {
	kernel_1,
	kernel_2,
};

void main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(arg->no_of_nodes, (vx_spawn_tasks_cb)sc_tests[arg->testid], arg);
}
