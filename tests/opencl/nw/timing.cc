#include <stdio.h>

#include "timing.h"

void time_measure_start(struct timeval *tv)
{
	gettimeofday(tv, NULL);
}

void time_measure_end(struct timeval *tv)
{
	struct timeval tv_now, tv_diff;
	double d;

	gettimeofday(&tv_now, NULL);
	tvsub(&tv_now, tv, &tv_diff);

	d = (double) tv_diff.tv_sec * 1000.0 + (double) tv_diff.tv_usec / 1000.0;
	printf("Time (Memory Copy and Launch) = %f (ms)\n", d);
}

float probe_event_time(cl_event event, cl_command_queue command_queue) {
    cl_int error=0;
    cl_ulong eventStart,eventEnd;
    clFinish(command_queue);
    error = clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,
                                    sizeof(cl_ulong),&eventStart,NULL);
    if (error != CL_SUCCESS) {
        printf("ERROR (%d) in event start profiling.\n", error);
        return 0;
    }
    error = clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,
                                    sizeof(cl_ulong),&eventEnd,NULL);
    if (error != CL_SUCCESS) {
        printf("ERROR (%d) in event end profiling.\n", error);
        return 0;
    }

    return (float)((eventEnd-eventStart)/1000000.0);
}
