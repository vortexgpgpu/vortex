#ifndef COMMON_H
#define COMMON_H

// Value work-item gid (local id lid) writes in round r of the exchange kernels;
// the host recomputes it to check what each work-item read. The work grows with
// lid, so later warps write later and a reader that does not wait for its peer
// reads the previous round's value.
static inline unsigned int fence_value(unsigned int gid, unsigned int lid, unsigned int r) {
  unsigned int v = gid * 2654435761u + r * 40503u + 1u;
  for (unsigned int i = 0; i < lid * 8u; ++i) {
    v = v * 1103515245u + 12345u;
  }
  return v;
}

// Peer whose value work-item lid reads in round r: half a work-group away, so
// in another warp in round 0 whenever the work-group spans more than one.
#define FENCE_PEER(lid, lsz, r) (((lid) + (lsz) / 2 + (r)) % (lsz))

#endif // COMMON_H
