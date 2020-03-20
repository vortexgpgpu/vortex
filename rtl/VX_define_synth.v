
`ifndef VX_DEFINE_SYNTH
`define VX_DEFINE_SYNTH

`define NT 4
`define NW 8
`define NUMBER_CORES_PER_CLUSTER 2
`define NUMBER_CLUSTERS 1
`define DCACHE_SIZE_BYTES 4096
`define ICACHE_SIZE_BYTES 1024

// L2 Cache size
`define LLCACHE_SIZE_BYTES 8192


// `define QUEUE_FORCE_MLAB 1

// Use l3 cache (required for cluster behavior)
// `define L3C 1

`endif
