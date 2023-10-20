/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifndef __GPUINFOH__
#define __GPUINFOH__

#ifdef __cplusplus
extern "C" {
#endif

void compute_active_thread(size_t *thread,
			   size_t *grid,
			   int task,
			   int pad,
			   int major,
			   int minor,
			   int sm);

#ifdef __cplusplus
}
#endif

#endif
