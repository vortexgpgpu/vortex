// B+ tree search kernels (Rodinia b+tree) — standalone Vortex port.
//
// One work-group processes one query; ORDER work-items cooperate over a single
// tree node (each thread owns one key/index slot). The traversal logic is kept
// identical to the original benchmark. ORDER is fixed small so the work-group
// size (== ORDER) never exceeds the device max (NUM_WARPS*NUM_THREADS = 16).

#ifndef ORDER
#define ORDER 16
#endif

// Record to which a leaf key refers.
typedef struct record {
  int value;
} record;

// Flattened B+ tree node. keys[]/indices[] are padded to ORDER+1 with sentinel
// keys (INT_MIN at slot 0, INT_MAX beyond the last real key) so the interval
// test below selects exactly one slot. is_leaf is unused by the traversal but
// kept so the host/device struct layouts match byte-for-byte.
typedef struct knode {
  int location;
  int indices[ORDER + 1];
  int keys[ORDER + 1];
  bool is_leaf;
  int num_keys;
} knode;

//========================================================================
//  findK: point query — descend to the leaf and fetch the matching record.
//========================================================================
__kernel void findK(long height,
                    __global knode *knodesD,
                    long knodes_elem,
                    __global record *recordsD,
                    __global long *currKnodeD,
                    __global long *offsetD,
                    __global int *keysD,
                    __global record *ansD) {
  int thid = get_local_id(0);
  int bid = get_group_id(0);

  // Walk the tree one level per iteration.
  int i;
  for (i = 0; i < height; i++) {
    // If the search key falls in this thread's key interval, descend.
    if ((knodesD[currKnodeD[bid]].keys[thid]) <= keysD[bid] && (knodesD[currKnodeD[bid]].keys[thid + 1] > keysD[bid])) {
      if (knodesD[offsetD[bid]].indices[thid] < knodes_elem) {
        offsetD[bid] = knodesD[offsetD[bid]].indices[thid];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // Advance to the chosen child for the next level.
    if (thid == 0) {
      currKnodeD[bid] = offsetD[bid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Candidate leaf reached: if this thread's key matches, return the record.
  if (knodesD[currKnodeD[bid]].keys[thid] == keysD[bid]) {
    ansD[bid].value = recordsD[knodesD[currKnodeD[bid]].indices[thid]].value;
  }
}

//========================================================================
//  findRangeK: range query — locate the record indices of the start and end
//  keys (each descended independently), returning start index and length.
//========================================================================
__kernel void findRangeK(long height,
                         __global knode *knodesD,
                         long knodes_elem,
                         __global long *currKnodeD,
                         __global long *offsetD,
                         __global long *lastKnodeD,
                         __global long *offset_2D,
                         __global int *startD,
                         __global int *endD,
                         __global int *RecstartD,
                         __global int *ReclenD) {
  int thid = get_local_id(0);
  int bid = get_group_id(0);

  int i;
  for (i = 0; i < height; i++) {
    if ((knodesD[currKnodeD[bid]].keys[thid] <= startD[bid]) && (knodesD[currKnodeD[bid]].keys[thid + 1] > startD[bid])) {
      if (knodesD[currKnodeD[bid]].indices[thid] < knodes_elem) {
        offsetD[bid] = knodesD[currKnodeD[bid]].indices[thid];
      }
    }
    if ((knodesD[lastKnodeD[bid]].keys[thid] <= endD[bid]) && (knodesD[lastKnodeD[bid]].keys[thid + 1] > endD[bid])) {
      if (knodesD[lastKnodeD[bid]].indices[thid] < knodes_elem) {
        offset_2D[bid] = knodesD[lastKnodeD[bid]].indices[thid];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thid == 0) {
      currKnodeD[bid] = offsetD[bid];
      lastKnodeD[bid] = offset_2D[bid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Record index of the starting key.
  if (knodesD[currKnodeD[bid]].keys[thid] == startD[bid]) {
    RecstartD[bid] = knodesD[currKnodeD[bid]].indices[thid];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Record count spanned by [start, end].
  if (knodesD[lastKnodeD[bid]].keys[thid] == endD[bid]) {
    ReclenD[bid] = knodesD[lastKnodeD[bid]].indices[thid] - RecstartD[bid] + 1;
  }
}
