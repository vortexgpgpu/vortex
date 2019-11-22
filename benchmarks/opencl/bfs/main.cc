//--by Jianbin Fang

#define __CL_ENABLE_EXCEPTIONS
#include <cstdlib>
#include <iostream>
#include <string>
#include <cstring>

#ifdef  PROFILING
#include "timer.h"
#endif

#include "CLHelper.h"
#include "util.h"

#define MAX_THREADS_PER_BLOCK 256

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};


//----------------------------------------------------------
//--bfs on cpu
//--programmer:	jianbin
//--date:	26/01/2011
//--note: width is changed to the new_width
//----------------------------------------------------------
void run_bfs_cpu(int no_of_nodes, Node *h_graph_nodes, int edge_list_size, \
		int *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask, \
		char *h_graph_visited, int *h_cost_ref){
	char stop;
	int k = 0;
	do{
		//if no thread changes this value then the loop stops
		stop=false;
		for(int tid = 0; tid < no_of_nodes; tid++ )
		{
			if (h_graph_mask[tid] == true){ 
				h_graph_mask[tid]=false;
				for(int i=h_graph_nodes[tid].starting; i<(h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting); i++){
					int id = h_graph_edges[i];	//--cambine: node id is connected with node tid
					if(!h_graph_visited[id]){	//--cambine: if node id has not been visited, enter the body below
						h_cost_ref[id]=h_cost_ref[tid]+1;
						h_updating_graph_mask[id]=true;
					}
				}
			}		
		}

  		for(int tid=0; tid< no_of_nodes ; tid++ )
		{
			if (h_updating_graph_mask[tid] == true){
			h_graph_mask[tid]=true;
			h_graph_visited[tid]=true;
			stop=true;
			h_updating_graph_mask[tid]=false;
			}
		}
		k++;
	}
	while(stop);
}
//----------------------------------------------------------
//--breadth first search on GPUs
//----------------------------------------------------------
void run_bfs_gpu(int no_of_nodes, Node *h_graph_nodes, int edge_list_size, \
		int *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask, \
		char *h_graph_visited, int *h_cost) 
					throw(std::string){

	//int number_elements = height*width;
	char h_over;
	cl_mem d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, \
			d_graph_visited, d_cost, d_over;
	try{
		//--1 transfer data from host to device
		_clInit();			
		d_graph_nodes = _clMalloc(no_of_nodes*sizeof(Node), h_graph_nodes);
		d_graph_edges = _clMalloc(edge_list_size*sizeof(int), h_graph_edges);
		d_graph_mask = _clMallocRW(no_of_nodes*sizeof(char), h_graph_mask);
		d_updating_graph_mask = _clMallocRW(no_of_nodes*sizeof(char), h_updating_graph_mask);
		d_graph_visited = _clMallocRW(no_of_nodes*sizeof(char), h_graph_visited);


		d_cost = _clMallocRW(no_of_nodes*sizeof(int), h_cost);
		d_over = _clMallocRW(sizeof(char), &h_over);
		
		_clMemcpyH2D(d_graph_nodes, no_of_nodes*sizeof(Node), h_graph_nodes);
		_clMemcpyH2D(d_graph_edges, edge_list_size*sizeof(int), h_graph_edges);	
		_clMemcpyH2D(d_graph_mask, no_of_nodes*sizeof(char), h_graph_mask);	
		_clMemcpyH2D(d_updating_graph_mask, no_of_nodes*sizeof(char), h_updating_graph_mask);	
		_clMemcpyH2D(d_graph_visited, no_of_nodes*sizeof(char), h_graph_visited);	
		_clMemcpyH2D(d_cost, no_of_nodes*sizeof(int), h_cost);	
			
		//--2 invoke kernel
#ifdef	PROFILING
		timer kernel_timer;
		double kernel_time = 0.0;		
		kernel_timer.reset();
		kernel_timer.start();
#endif
		do{
			h_over = false;
			_clMemcpyH2D(d_over, sizeof(char), &h_over);
			//--kernel 0
			int kernel_id = 0;
			int kernel_idx = 0;
			_clSetArgs(kernel_id, kernel_idx++, d_graph_nodes);
			_clSetArgs(kernel_id, kernel_idx++, d_graph_edges);
			_clSetArgs(kernel_id, kernel_idx++, d_graph_mask);
			_clSetArgs(kernel_id, kernel_idx++, d_updating_graph_mask);
			_clSetArgs(kernel_id, kernel_idx++, d_graph_visited);
			_clSetArgs(kernel_id, kernel_idx++, d_cost);
			_clSetArgs(kernel_id, kernel_idx++, &no_of_nodes, sizeof(int));
			
			//int work_items = no_of_nodes;
			_clInvokeKernel(kernel_id, no_of_nodes, work_group_size);
			
			//--kernel 1
			kernel_id = 1;
			kernel_idx = 0;			
			_clSetArgs(kernel_id, kernel_idx++, d_graph_mask);
			_clSetArgs(kernel_id, kernel_idx++, d_updating_graph_mask);
			_clSetArgs(kernel_id, kernel_idx++, d_graph_visited);
			_clSetArgs(kernel_id, kernel_idx++, d_over);
			_clSetArgs(kernel_id, kernel_idx++, &no_of_nodes, sizeof(int));
			
			//work_items = no_of_nodes;
			_clInvokeKernel(kernel_id, no_of_nodes, work_group_size);			
			
			_clMemcpyD2H(d_over,sizeof(char), &h_over);
			}while(h_over);
			
		_clFinish();
#ifdef	PROFILING
		kernel_timer.stop();
		kernel_time = kernel_timer.getTimeInSeconds();
#endif
		//--3 transfer data from device to host
		_clMemcpyD2H(d_cost,no_of_nodes*sizeof(int), h_cost);
		//--statistics
#ifdef	PROFILING
		std::cout<<"kernel time(s):"<<kernel_time<<std::endl;		
#endif
		//--4 release cl resources.
		_clFree(d_graph_nodes);
		_clFree(d_graph_edges);
		_clFree(d_graph_mask);
		_clFree(d_updating_graph_mask);
		_clFree(d_graph_visited);
		_clFree(d_cost);
		_clFree(d_over);
		_clRelease();
	}
	catch(std::string msg){		
		_clFree(d_graph_nodes);
		_clFree(d_graph_edges);
		_clFree(d_graph_mask);
		_clFree(d_updating_graph_mask);
		_clFree(d_graph_visited);
		_clFree(d_cost);
		_clFree(d_over);
		_clRelease();
		std::string e_str = "in run_transpose_gpu -> ";
		e_str += msg;
		throw(e_str);
	}
	return ;
}
void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);

}
//----------------------------------------------------------
//--cambine:	main function
//--author:		created by Jianbin Fang
//--date:		25/01/2011
//----------------------------------------------------------
int main(int argc, char * argv[])
{
	int no_of_nodes;
	int edge_list_size;
	FILE *fp;
	Node* h_graph_nodes;
	char *h_graph_mask, *h_updating_graph_mask, *h_graph_visited;
	try{
		char *input_f;
		if(argc!=2){
		  Usage(argc, argv);
		  exit(0);
		}
	
		input_f = argv[1];
		printf("Reading File\n");
		//Read in Graph from a file
		fp = fopen(input_f,"r");
		if(!fp){
		  printf("Error Reading graph file\n");
		  return 0;
		}

		int source = 0;

		fscanf(fp,"%d",&no_of_nodes);

		int num_of_blocks = 1;
		int num_of_threads_per_block = no_of_nodes;

		//Make execution Parameters according to the number of nodes
		//Distribute threads across multiple Blocks if necessary
		if(no_of_nodes>MAX_THREADS_PER_BLOCK){
			num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
			num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
		}
		work_group_size = num_of_threads_per_block;
		// allocate host memory
		h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
		h_graph_mask = (char*) malloc(sizeof(char)*no_of_nodes);
		h_updating_graph_mask = (char*) malloc(sizeof(char)*no_of_nodes);
		h_graph_visited = (char*) malloc(sizeof(char)*no_of_nodes);
	
		int start, edgeno;   
		// initalize the memory
		for(int i = 0; i < no_of_nodes; i++){
			fscanf(fp,"%d %d",&start,&edgeno);
			h_graph_nodes[i].starting = start;
			h_graph_nodes[i].no_of_edges = edgeno;
			h_graph_mask[i]=false;
			h_updating_graph_mask[i]=false;
			h_graph_visited[i]=false;
		}
		//read the source node from the file
		fscanf(fp,"%d",&source);
		source=0;
		//set the source node as true in the mask
		h_graph_mask[source]=true;
		h_graph_visited[source]=true;
    	fscanf(fp,"%d",&edge_list_size);
   		int id,cost;
		int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
		for(int i=0; i < edge_list_size ; i++){
			fscanf(fp,"%d",&id);
			fscanf(fp,"%d",&cost);
			h_graph_edges[i] = id;
		}

		if(fp)
			fclose(fp);    
		// allocate mem for the result on host side
		int	*h_cost = (int*) malloc(sizeof(int)*no_of_nodes);
		int *h_cost_ref = (int*)malloc(sizeof(int)*no_of_nodes);
		for(int i=0;i<no_of_nodes;i++){
			h_cost[i]=-1;
			h_cost_ref[i] = -1;
		}
		h_cost[source]=0;
		h_cost_ref[source]=0;		
		//---------------------------------------------------------
		//--gpu entry
		run_bfs_gpu(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost);	
		//---------------------------------------------------------
		//--cpu entry
		// initalize the memory again
		for(int i = 0; i < no_of_nodes; i++){
			h_graph_mask[i]=false;
			h_updating_graph_mask[i]=false;
			h_graph_visited[i]=false;
		}
		//set the source node as true in the mask
		source=0;
		h_graph_mask[source]=true;
		h_graph_visited[source]=true;
		run_bfs_cpu(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost_ref);
		//---------------------------------------------------------
		//--result varification
		compare_results<int>(h_cost_ref, h_cost, no_of_nodes);
		//release host memory		
		free(h_graph_nodes);
		free(h_graph_mask);
		free(h_updating_graph_mask);
		free(h_graph_visited);

	}
	catch(std::string msg){
		std::cout<<"--cambine: exception in main ->"<<msg<<std::endl;
		//release host memory
		free(h_graph_nodes);
		free(h_graph_mask);
		free(h_updating_graph_mask);
		free(h_graph_visited);		
	}
		
    return 0;
}
