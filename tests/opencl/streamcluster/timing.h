#ifndef __TIMING_H__
#define __TIMING_H__

#include <sys/time.h>
#include <CL/cl.h>

void time_measure_start(struct timeval *tv);
void time_measure_end(struct timeval *tv);

/* tvsub: ret = x - y. */
static inline void tvsub(struct timeval *x,
						 struct timeval *y,
						 struct timeval *ret)
{
	ret->tv_sec = x->tv_sec - y->tv_sec;
	ret->tv_usec = x->tv_usec - y->tv_usec;
	if (ret->tv_usec < 0) {
		ret->tv_sec--;
		ret->tv_usec += 1000000;
	}
}

float probe_event_time(cl_event, cl_command_queue);

#endif
