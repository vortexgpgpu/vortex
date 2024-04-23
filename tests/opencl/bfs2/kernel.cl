/* ============================================================
//--cambine: kernel funtion of Breadth-First-Search
//--author:	created by Jianbin Fang
//--date:	06/12/2010
============================================================ */
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable
//Structure to hold a node information
typedef struct{
	int starting;
	int no_of_edges;
} Node;
//--7 parameters
__kernel void BFS_1( const __global Node* g_graph_nodes,
					const __global int* g_graph_edges, 
					__global char* g_graph_mask, 
					__global char* g_updating_graph_mask, 
					__global char* g_graph_visited, 
					__global int* g_cost, 
					const  int no_of_nodes){
	int tid = get_global_id(0);
	if( tid<no_of_nodes && g_graph_mask[tid]){
		g_graph_mask[tid]=false;
		for(int i=g_graph_nodes[tid].starting; i<(g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++){
			int id = g_graph_edges[i];
			if(!g_graph_visited[id]){
				g_cost[id]=g_cost[tid]+1;
				g_updating_graph_mask[id]=true;
				}
			}
	}	
}

//--5 parameters
__kernel void BFS_2(__global char* g_graph_mask, 
					__global char* g_updating_graph_mask, 
					__global char* g_graph_visited, 
					__global char* g_over,
					const  int no_of_nodes
					) {
	int tid = get_global_id(0);
	if( tid<no_of_nodes && g_updating_graph_mask[tid]){

		g_graph_mask[tid]=true;
		g_graph_visited[tid]=true;
		*g_over=true;
		g_updating_graph_mask[tid]=false;
	}
}


