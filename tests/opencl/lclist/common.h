#ifndef COMMON_H
#define COMMON_H

// Pre-allocated node pool layout: node 0 and node 1 are sentinels, the
// remaining nodes are owned one-per-work-item (work-item i owns node i+2).
#define HEAD_NODE 0   // sentinel, key = INT_MIN
#define TAIL_NODE 1   // sentinel, key = INT_MAX
#define FIRST_DATA_NODE 2

#endif // COMMON_H
