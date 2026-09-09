#ifndef SOFTWARE
#define SOFTWARE 0
#endif
#ifndef BENCH
#define BENCH 0
#endif

inline int take(volatile __global int* lock) {
  int won = atomic_xchg(lock, 1) == 0;
  if (won) {
    mem_fence(CLK_GLOBAL_MEM_FENCE);
  }
  return won;
}

inline void acquire(volatile __global int* lock) {
  while (!take(lock)) {}
}

inline void release(volatile __global int* lock) {
  mem_fence(CLK_GLOBAL_MEM_FENCE);
  atomic_xchg(lock, 0);
}

// Shared arguments keep the host and parameter sweeps identical across variants.
__kernel void evaluate(__global int* data, __global int* links,
                       __global int* locks, __global int* output,
                       __global int* audit, int n, int resources,
                       int rounds, int percent, int stride) {
  int gid = get_global_id(0);
#if SOFTWARE
  // Software baseline: serialize complete synchronization operations within a
  // work-group. A lone active lane can acquire and release a lock before the
  // IPDOM stack reaches the enclosing branch reconvergence point.
  for (int sw_turn = 0; sw_turn < get_local_size(0); ++sw_turn) {
    if (get_local_id(0) == sw_turn) {
#endif
#if BENCH <= 4
  int bucket = (gid / 2) % resources;
  for (int iter = 0; iter < rounds; ++iter) {
    int value = (gid * 73 + iter * 19) % 100003;
    int enter = BENCH < 2 || BENCH == 4 || (gid < (n * percent) / 100);
    if (enter) {
#if SOFTWARE
      int done = 0;
      while (!done) {
        if (take(&locks[bucket * stride])) {
#else
      acquire(&locks[bucket * stride]);
#endif
#if BENCH < 2
          data[bucket] += 1;
#elif BENCH < 4
          data[bucket] = max(data[bucket], value);
#else
          int node = iter * n + gid;
          links[node] = data[bucket];
          data[bucket] = node;
#endif
          release(&locks[bucket * stride]);
#if SOFTWARE
          done = 1;
        }
      }
#endif
    }
  }
#elif BENCH == 5
  // Transfers hold the lower account lock before acquiring the higher one.
  int a = (gid / 2) % resources;
  int b = (a + 1 + gid % (resources - 1)) % resources;
  int lo = min(a, b), hi = max(a, b);
  for (int iter = 0; iter < rounds; ++iter) {
#if SOFTWARE
    int state = 0;
    while (state != 2) {
      if (state == 0) {
        if (take(&locks[lo * stride])) {
          state = 1;
        }
      } else if (take(&locks[hi * stride])) {
#else
    acquire(&locks[lo * stride]);
    acquire(&locks[hi * stride]);
#endif
        data[a] -= 1;
        data[b] += 1;
        release(&locks[hi * stride]);
        release(&locks[lo * stride]);
#if SOFTWARE
        state = 2;
      }
    }
#endif
  }
#elif BENCH == 6
  // Preallocated sorted nodes avoid introducing a memory-reclamation policy.
  int key = 1 + (gid * 17) % resources;
  int pred = 0, curr = 0;
  int update = gid < (n * percent) / 100;
#if SOFTWARE
  int state = 0;
  while (state != 3) {
    if (state == 0) {
      if (take(&locks[0])) {
        curr = links[0];
        state = 1;
      }
    } else if (state == 1) {
      if (take(&locks[curr * stride])) {
        state = 2;
      }
    } else {
      if (data[curr] < key) {
        release(&locks[pred * stride]);
        pred = curr;
        curr = links[curr];
        state = 1;
      } else {
        output[gid] = data[curr] == key;
        if (update) {
          audit[curr] += 1;
        }
        release(&locks[curr * stride]);
        release(&locks[pred * stride]);
        state = 3;
      }
    }
  }
#else
  acquire(&locks[0]);
  curr = links[0];
  acquire(&locks[curr * stride]);
  while (data[curr] < key) {
    release(&locks[pred * stride]);
    pred = curr;
    curr = links[curr];
    acquire(&locks[curr * stride]);
  }
  output[gid] = data[curr] == key;
  if (update) {
    audit[curr] += 1;
  }
  release(&locks[curr * stride]);
  release(&locks[pred * stride]);
#endif
#elif BENCH == 7
  // Grid distance constraints; replay the lock-serialized updates on the host.
  int side = (int)sqrt((float)resources);
  int horizontal = side * (side - 1);
  int edge = gid % (2 * horizontal);
  int e = edge % horizontal;
  int a = edge < horizontal ? (e / (side - 1)) * side + e % (side - 1) : e;
  int b = a + (edge < horizontal ? 1 : side);
  for (int iter = 0; iter < rounds; ++iter) {
#if SOFTWARE
    int state = 0;
    while (state != 2) {
      if (state == 0) {
        if (take(&locks[a * stride])) {
          state = 1;
        }
      } else if (take(&locks[b * stride])) {
#else
    acquire(&locks[a * stride]);
    acquire(&locks[b * stride]);
#endif
        int ticket = atomic_inc(&audit[0]);
        audit[ticket + 1] = edge;
        float dx = (float)(data[2 * b] - data[2 * a]);
        float dy = (float)(data[2 * b + 1] - data[2 * a + 1]);
        float distance = sqrt(dx * dx + dy * dy);
        float scale = distance > 0 ? 0.5f * (distance - 1024.0f) / distance : 0;
        int cx = (int)(dx * scale), cy = (int)(dy * scale);
        data[2 * a] += cx;
        data[2 * a + 1] += cy;
        data[2 * b] -= cx;
        data[2 * b + 1] -= cy;
        release(&locks[b * stride]);
        release(&locks[a * stride]);
#if SOFTWARE
        state = 2;
      }
    }
#endif
  }
#elif BENCH == 8
  // Heap-ordered octree: parent publishes child output offsets.
  int node = gid + 1;
  int parent = node == 1 ? 0 : (node - 2) / 8 + 1;
#if SOFTWARE
  // The software counterpart computes the same offset by walking to the root,
  // avoiding the producer/consumer wait that stack-based IPDOM cannot run.
  int offset = 0;
  for (int cursor = node; cursor > 1; cursor = (cursor - 2) / 8 + 1) {
    int ancestor = (cursor - 2) / 8 + 1;
    for (int sibling = 8 * ancestor - 6; sibling < cursor; ++sibling) {
      offset += data[sibling];
    }
  }
  output[node] = offset;
  atomic_xchg(&locks[node], 1);
#else
  while (!atomic_add(&locks[parent], 0)) {}
  mem_fence(CLK_GLOBAL_MEM_FENCE);
  int offset = output[parent];
  if (node > 1) {
    for (int sibling = 8 * parent - 6; sibling < node; ++sibling) {
      offset += data[sibling];
    }
  }
  output[node] = offset;
  mem_fence(CLK_GLOBAL_MEM_FENCE);
  atomic_xchg(&locks[node], 1);
#endif
#endif
#if SOFTWARE
    }
  }
#endif
}
