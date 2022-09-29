#ifndef _COMMON_H_

#define _COMMON_H_

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

typedef struct{
int starting;
int no_of_edges;
} Node;

typedef struct {
uint32_t testid;
uint32_t hover_addr;
uint32_t graphnodes_addr;
uint32_t graphedges_addr;
uint32_t graphmask_addr;
uint32_t graphupmask_addr;
uint32_t graphvisited_addr;
uint32_t gcost_addr;
uint32_t no_of_nodes;
} kernel_arg_t;

#endif
