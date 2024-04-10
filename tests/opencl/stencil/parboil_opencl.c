/*
 * (c) 2007 The Board of Trustees of the University of Illinois.
 */

#include <parboil.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <CL/cl.h>

#if _POSIX_VERSION >= 200112L
# include <sys/time.h>
#endif

//#include "perfmon.h"

cl_context *clContextPtr;
cl_command_queue *clCommandQueuePtr;

// #define DISABLE_PARBOIL_TIMER

/*****************************************************************************/
/* Timer routines */

static int is_async(enum pb_TimerID timer)
{
  return (timer == pb_TimerID_KERNEL) || 
             (timer == pb_TimerID_COPY_ASYNC);
}

static int is_blocking(enum pb_TimerID timer)
{
  return (timer == pb_TimerID_COPY) || (timer == pb_TimerID_NONE);
}

#define INVALID_TIMERID pb_TimerID_LAST

static int asyncs_outstanding(struct pb_TimerSet* timers)
{
  return (timers->async_markers != NULL) && 
           (timers->async_markers->timerID != INVALID_TIMERID);
}

static struct pb_async_time_marker_list * 
get_last_async(struct pb_TimerSet* timers)
{
  /* Find the last event recorded thus far */
  struct pb_async_time_marker_list * last_event = timers->async_markers;
  if(last_event != NULL && last_event->timerID != INVALID_TIMERID) {
    while(last_event->next != NULL && 
            last_event->next->timerID != INVALID_TIMERID)
      last_event = last_event->next;
    return last_event;
  } else
    return NULL;
}

static void insert_marker(struct pb_TimerSet* tset, enum pb_TimerID timer)
{
  cl_int ciErrNum = CL_SUCCESS;
  struct pb_async_time_marker_list ** new_event = &(tset->async_markers);

  while(*new_event != NULL && (*new_event)->timerID != INVALID_TIMERID) {
    new_event = &((*new_event)->next);
  }

  if(*new_event == NULL) {
    *new_event = (struct pb_async_time_marker_list *) 
      			malloc(sizeof(struct pb_async_time_marker_list));
    (*new_event)->marker = calloc(1, sizeof(cl_event));
    /*
    // I don't think this is needed at all. I believe clEnqueueMarker 'creates' the event
#if ( __OPENCL_VERSION__ >= CL_VERSION_1_1 )
fprintf(stderr, "Creating Marker [%d]\n", timer);
    *((cl_event *)((*new_event)->marker)) = clCreateUserEvent(*clContextPtr, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
      fprintf(stderr, "Error Creating User Event Object!\n");
    }
    ciErrNum = clSetUserEventStatus(*((cl_event *)((*new_event)->marker)), CL_QUEUED);
    if (ciErrNum != CL_SUCCESS) {
      fprintf(stderr, "Error Setting User Event Status!\n");
    }
#endif
*/
    (*new_event)->next = NULL;
  }

  /* valid event handle now aquired: insert the event record */
  (*new_event)->label = NULL;
  (*new_event)->timerID = timer;
  ciErrNum = clEnqueueMarker(*clCommandQueuePtr, (cl_event *)(*new_event)->marker);
  if (ciErrNum != CL_SUCCESS) {
      fprintf(stderr, "Error Enqueueing Marker!\n");
  }

}

static void insert_submarker(struct pb_TimerSet* tset, char *label, enum pb_TimerID timer)
{
  cl_int ciErrNum = CL_SUCCESS;
  struct pb_async_time_marker_list ** new_event = &(tset->async_markers);

  while(*new_event != NULL && (*new_event)->timerID != INVALID_TIMERID) {
    new_event = &((*new_event)->next);
  }

  if(*new_event == NULL) {
    *new_event = (struct pb_async_time_marker_list *) 
      			malloc(sizeof(struct pb_async_time_marker_list));
    (*new_event)->marker = calloc(1, sizeof(cl_event));
    /*
#if ( __OPENCL_VERSION__ >= CL_VERSION_1_1 )
fprintf(stderr, "Creating SubMarker %s[%d]\n", label, timer);
    *((cl_event *)((*new_event)->marker)) = clCreateUserEvent(*clContextPtr, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
      fprintf(stderr, "Error Creating User Event Object!\n");
    }
    ciErrNum = clSetUserEventStatus(*((cl_event *)((*new_event)->marker)), CL_QUEUED);
    if (ciErrNum != CL_SUCCESS) {
      fprintf(stderr, "Error Setting User Event Status!\n");
    }
#endif
*/
    (*new_event)->next = NULL;
  }

  /* valid event handle now aquired: insert the event record */
  (*new_event)->label = label;
  (*new_event)->timerID = timer;
  ciErrNum = clEnqueueMarker(*clCommandQueuePtr, (cl_event *)(*new_event)->marker);
  if (ciErrNum != CL_SUCCESS) {
      fprintf(stderr, "Error Enqueueing Marker!\n");
  }

}


/* Assumes that all recorded events have completed */
static pb_Timestamp record_async_times(struct pb_TimerSet* tset)
{
  struct pb_async_time_marker_list * next_interval = NULL;
  struct pb_async_time_marker_list * last_marker = get_last_async(tset);
  pb_Timestamp total_async_time = 0;
  enum pb_TimerID timer;
  
  for(next_interval = tset->async_markers; next_interval != last_marker; 
      next_interval = next_interval->next) {
    cl_ulong command_start=0, command_end=0;
    cl_int ciErrNum = CL_SUCCESS;
    
    ciErrNum = clGetEventProfilingInfo(*((cl_event *)next_interval->marker), CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &command_start, NULL);
    if (ciErrNum != CL_SUCCESS) {
      fprintf(stderr, "Error getting first EventProfilingInfo: %d\n", ciErrNum);
    }

    ciErrNum = clGetEventProfilingInfo(*((cl_event *)next_interval->next->marker), CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &command_end, NULL);
    if (ciErrNum != CL_SUCCESS) {
      fprintf(stderr, "Error getting second EventProfilingInfo: %d\n", ciErrNum);
    } 
    
    pb_Timestamp interval = (pb_Timestamp) (((double)(command_end - command_start)) / 1e3);
    tset->timers[next_interval->timerID].elapsed += interval;
    if (next_interval->label != NULL) {
      struct pb_SubTimer *subtimer = tset->sub_timer_list[next_interval->timerID]->subtimer_list;
      while (subtimer != NULL) {
        if ( strcmp(subtimer->label, next_interval->label) == 0) {
          subtimer->timer.elapsed += interval;
          break;
        }
        subtimer = subtimer->next;
      }      
    }        
    total_async_time += interval;
    next_interval->timerID = INVALID_TIMERID;
  }

  if(next_interval != NULL)
    next_interval->timerID = INVALID_TIMERID;
  
  return total_async_time;
}

static void
accumulate_time(pb_Timestamp *accum,
		pb_Timestamp start,
		pb_Timestamp end)
{
//#if _POSIX_VERSION >= 200112L
  *accum += end - start;
//#else
//# error "Timestamps not implemented for this system"
//#endif
}

//#if _POSIX_VERSION >= 200112L
static pb_Timestamp get_time()
{
  //struct timeval tv;
  //gettimeofday(&tv, NULL);
  //return (pb_Timestamp) (tv.tv_sec * 1000000LL + tv.tv_usec);
  return 0;
}
//#else
//# error "no supported time libraries are available on this platform"
//#endif

void
pb_ResetTimer(struct pb_Timer *timer)
{
//#ifndef DISABLE_PARBOIL_TIMER
  timer->state = pb_Timer_STOPPED;

//#if _POSIX_VERSION >= 200112L
  timer->elapsed = 0;
//#else
//# error "pb_ResetTimer: not implemented for this system"
//#endif
//#endif
}

void
pb_StartTimer(struct pb_Timer *timer)
{
/*#ifndef DISABLE_PARBOIL_TIMER
  if (timer->state != pb_Timer_STOPPED) {
    fputs("Ignoring attempt to start a running timer\n", stderr);
    return;
  }

  timer->state = pb_Timer_RUNNING;

#if _POSIX_VERSION >= 200112L
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    timer->init = tv.tv_sec * 1000000LL + tv.tv_usec;
  }
#else
# error "pb_StartTimer: not implemented for this system"
#endif
#endif*/
}

void
pb_StartTimerAndSubTimer(struct pb_Timer *timer, struct pb_Timer *subtimer)
{
/*#ifndef DISABLE_PARBOIL_TIMER

  unsigned int numNotStopped = 0x3; // 11
  if (timer->state != pb_Timer_STOPPED) {
    fputs("Warning: Timer was not stopped\n", stderr);
    numNotStopped &= 0x1; // Zero out 2^1
  }
  if (subtimer->state != pb_Timer_STOPPED) {
    fputs("Warning: Subtimer was not stopped\n", stderr);
    numNotStopped &= 0x2; // Zero out 2^0
  }
  if (numNotStopped == 0x0) {
    fputs("Ignoring attempt to start running timer and subtimer\n", stderr);
    return;
  }

  timer->state = pb_Timer_RUNNING;
  subtimer->state = pb_Timer_RUNNING;

#if _POSIX_VERSION >= 200112L
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    
    if (numNotStopped & 0x2) {
      timer->init = tv.tv_sec * 1000000LL + tv.tv_usec;
    }
  
    if (numNotStopped & 0x1) {
      subtimer->init = tv.tv_sec * 1000000LL + tv.tv_usec;
    }
  }
#else
# error "pb_StartTimer: not implemented for this system"
#endif

#endif*/
}

void
pb_StopTimer(struct pb_Timer *timer)
{
/*#ifndef DISABLE_PARBOIL_TIMER

  pb_Timestamp fini;

  if (timer->state != pb_Timer_RUNNING) {
    fputs("Ignoring attempt to stop a stopped timer\n", stderr);
    return;
  }

  timer->state = pb_Timer_STOPPED;

#if _POSIX_VERSION >= 200112L
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    fini = tv.tv_sec * 1000000LL + tv.tv_usec;
  }
#else
# error "pb_StopTimer: not implemented for this system"
#endif

  accumulate_time(&timer->elapsed, timer->init, fini);
  timer->init = fini;

#endif*/
}

void pb_StopTimerAndSubTimer(struct pb_Timer *timer, struct pb_Timer *subtimer) {
/*#ifndef DISABLE_PARBOIL_TIMER

  pb_Timestamp fini;

  unsigned int numNotRunning = 0x3; // 11
  if (timer->state != pb_Timer_RUNNING) {
    fputs("Warning: Timer was not running\n", stderr);
    numNotRunning &= 0x1; // Zero out 2^1
  }
  if (subtimer->state != pb_Timer_RUNNING) {
    fputs("Warning: Subtimer was not running\n", stderr);
    numNotRunning &= 0x2; // Zero out 2^0
  }
  if (numNotRunning == 0x0) {
    fputs("Ignoring attempt to stop stopped timer and subtimer\n", stderr);
    return;
  }


  timer->state = pb_Timer_STOPPED;
  subtimer->state = pb_Timer_STOPPED;

#if _POSIX_VERSION >= 200112L
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    fini = tv.tv_sec * 1000000LL + tv.tv_usec;
  }
#else
# error "pb_StopTimer: not implemented for this system"
#endif

  if (numNotRunning & 0x2) {
    accumulate_time(&timer->elapsed, timer->init, fini);
    timer->init = fini;
  }
  
  if (numNotRunning & 0x1) {
    accumulate_time(&subtimer->elapsed, subtimer->init, fini);
    subtimer->init = fini;
  }

#endif*/
}

/* Get the elapsed time in seconds. */
double
pb_GetElapsedTime(struct pb_Timer *timer)
{
  /*double ret;
#ifndef DISABLE_PARBOIL_TIMER

  if (timer->state != pb_Timer_STOPPED) {
    fputs("Elapsed time from a running timer is inaccurate\n", stderr);
  }

#if _POSIX_VERSION >= 200112L
  ret = timer->elapsed / 1e6;
#else
# error "pb_GetElapsedTime: not implemented for this system"
#endif
#endif
  return ret;*/
  return 0;
}

void
pb_InitializeTimerSet(struct pb_TimerSet *timers)
{
/*#ifndef DISABLE_PARBOIL_TIMER
  int n;

  timers->wall_begin = 0; //get_time();
  timers->current = pb_TimerID_NONE;

  timers->async_markers = NULL;
  
  for (n = 0; n < pb_TimerID_LAST; n++) {
    pb_ResetTimer(&timers->timers[n]);
    timers->sub_timer_list[n] = NULL;
  }
#endif*/
}

void pb_SetOpenCL(void *p_clContextPtr, void *p_clCommandQueuePtr) {
  clContextPtr = ((cl_context *)p_clContextPtr);
  clCommandQueuePtr = ((cl_command_queue *)p_clCommandQueuePtr);
}

void
pb_AddSubTimer(struct pb_TimerSet *timers, char *label, enum pb_TimerID pb_Category) {    
/*#ifndef DISABLE_PARBOIL_TIMER
  
  struct pb_SubTimer *subtimer = (struct pb_SubTimer *) malloc
    (sizeof(struct pb_SubTimer));
     
  int len = strlen(label);
    
  subtimer->label = (char *) malloc (sizeof(char)*(len+1));
  sprintf(subtimer->label, "%s\0", label);
  
  pb_ResetTimer(&subtimer->timer);
  subtimer->next = NULL;
  
  struct pb_SubTimerList *subtimerlist = timers->sub_timer_list[pb_Category];
  if (subtimerlist == NULL) {
    subtimerlist = (struct pb_SubTimerList *) calloc
      (1, sizeof(struct pb_SubTimerList));
    subtimerlist->subtimer_list = subtimer;
    timers->sub_timer_list[pb_Category] = subtimerlist;
  } else {
    // Append to list
    struct pb_SubTimer *element = subtimerlist->subtimer_list;
    while (element->next != NULL) {
      element = element->next;
    }
    element->next = subtimer;
  }
  
#endif*/
}

void
pb_SwitchToTimer(struct pb_TimerSet *timers, enum pb_TimerID timer)
{
#if 0
#ifndef DISABLE_PARBOIL_TIMER

  /* Stop the currently running timer */
  if (timers->current != pb_TimerID_NONE) {
    struct pb_SubTimerList *subtimerlist = timers->sub_timer_list[timers->current];
    struct pb_SubTimer *currSubTimer = (subtimerlist != NULL) ? subtimerlist->current : NULL;
  
    if (!is_async(timers->current) ) {
      if (timers->current != timer) {
        if (currSubTimer != NULL) {
          pb_StopTimerAndSubTimer(&timers->timers[timers->current], &currSubTimer->timer);
        } else {
          pb_StopTimer(&timers->timers[timers->current]);
        }
      } else {
        if (currSubTimer != NULL) {
          pb_StopTimer(&currSubTimer->timer);
        }
      }
    } else {
      insert_marker(timers, timer);
      if (!is_async(timer)) { // if switching to async too, keep driver going
        pb_StopTimer(&timers->timers[pb_TimerID_DRIVER]);
      }
    }
  }
  
  pb_Timestamp currentTime = 0; //get_time();

  /* The only cases we check for asynchronous task completion is 
   * when an overlapping CPU operation completes, or the next 
   * segment blocks on completion of previous async operations */
  if( asyncs_outstanding(timers) && 
      (!is_async(timers->current) || is_blocking(timer) ) ) {

    struct pb_async_time_marker_list * last_event = get_last_async(timers);
    /* CL_COMPLETE if completed */
    
    cl_int ciErrNum = CL_SUCCESS;
    cl_int async_done = CL_COMPLETE;
    
    ciErrNum = clGetEventInfo(*((cl_event *)last_event->marker), CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &async_done, NULL);
    if (ciErrNum != CL_SUCCESS) {
      fprintf(stderr, "Error Querying EventInfo!\n");
    }
    

    if(is_blocking(timer)) {
      /* Async operations completed after previous CPU operations: 
       * overlapped time is the total CPU time since this set of async 
       * operations were first issued */
       
      // timer to switch to is COPY or NONE 
      if(async_done != CL_COMPLETE) {
        accumulate_time(&(timers->timers[pb_TimerID_OVERLAP].elapsed), 
	                  timers->async_begin,currentTime);
      }	                  
	                  
      /* Wait on async operation completion */
      ciErrNum = clWaitForEvents(1, (cl_event *)last_event->marker);
      if (ciErrNum != CL_SUCCESS) {
        fprintf(stderr, "Error Waiting for Events!\n");
      }
      
      pb_Timestamp total_async_time = record_async_times(timers);

      /* Async operations completed before previous CPU operations: 
       * overlapped time is the total async time */
      if(async_done == CL_COMPLETE) {
        //fprintf(stderr, "Async_done: total_async_type = %lld\n", total_async_time);
        timers->timers[pb_TimerID_OVERLAP].elapsed += total_async_time;
      }

    } else 
    /* implies (!is_async(timers->current) && asyncs_outstanding(timers)) */
    // i.e. Current Not Async (not KERNEL/COPY_ASYNC) but there are outstanding
    // so something is deeper in stack
    if(async_done == CL_COMPLETE ) {
      /* Async operations completed before previous CPU operations: 
       * overlapped time is the total async time */
      timers->timers[pb_TimerID_OVERLAP].elapsed += record_async_times(timers);
    }   
  }

  /* Start the new timer */
  if (timer != pb_TimerID_NONE) {
    if(!is_async(timer)) {
      pb_StartTimer(&timers->timers[timer]);
    } else {
      // toSwitchTo Is Async (KERNEL/COPY_ASYNC)
      if (!asyncs_outstanding(timers)) {
        /* No asyncs outstanding, insert a fresh async marker */
      
        insert_marker(timers, timer);
        timers->async_begin = currentTime;
      } else if(!is_async(timers->current)) {
        /* Previous asyncs still in flight, but a previous SwitchTo
         * already marked the end of the most recent async operation, 
         * so we can rename that marker as the beginning of this async 
         * operation */
         
        struct pb_async_time_marker_list * last_event = get_last_async(timers);
        last_event->label = NULL;
        last_event->timerID = timer;
      }
      if (!is_async(timers->current)) {
        pb_StartTimer(&timers->timers[pb_TimerID_DRIVER]);
      }
    }
  }
  timers->current = timer;

#endif
#endif
}

void
pb_SwitchToSubTimer(struct pb_TimerSet *timers, char *label, enum pb_TimerID category) 
{
#ifndef DISABLE_PARBOIL_TIMER
  struct pb_SubTimerList *subtimerlist = timers->sub_timer_list[timers->current];
  struct pb_SubTimer *curr = (subtimerlist != NULL) ? subtimerlist->current : NULL;
  
  if (timers->current != pb_TimerID_NONE) {
    if (!is_async(timers->current) ) {
      if (timers->current != category) {
        if (curr != NULL) {
          pb_StopTimerAndSubTimer(&timers->timers[timers->current], &curr->timer);
        } else {
          pb_StopTimer(&timers->timers[timers->current]);
        }
      } else {
        if (curr != NULL) {
          pb_StopTimer(&curr->timer);
        }
      }
    } else {
      insert_submarker(timers, label, category);
      if (!is_async(category)) { // if switching to async too, keep driver going
        pb_StopTimer(&timers->timers[pb_TimerID_DRIVER]);
      }
    }
  }

  pb_Timestamp currentTime = 0; //get_time();

  /* The only cases we check for asynchronous task completion is 
   * when an overlapping CPU operation completes, or the next 
   * segment blocks on completion of previous async operations */
  if( asyncs_outstanding(timers) && 
      (!is_async(timers->current) || is_blocking(category) ) ) {

    struct pb_async_time_marker_list * last_event = get_last_async(timers);
    /* CL_COMPLETE if completed */

    cl_int ciErrNum = CL_SUCCESS;
    cl_int async_done = CL_COMPLETE;
    
    ciErrNum = clGetEventInfo(*((cl_event *)last_event->marker), CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &async_done, NULL);
    if (ciErrNum != CL_SUCCESS) {
      fprintf(stderr, "Error Querying EventInfo!\n");
    }

    if(is_blocking(category)) {
      /* Async operations completed after previous CPU operations: 
       * overlapped time is the total CPU time since this set of async 
       * operations were first issued */
       
      // timer to switch to is COPY or NONE 
      // if it hasn't already finished, then just take now and use that as the elapsed time in OVERLAP
      // anything happening after now isn't OVERLAP because everything is being stopped to wait for synchronization
      // it seems that the extra sync wall time isn't being recorded anywhere
      if(async_done != CL_COMPLETE) 
        accumulate_time(&(timers->timers[pb_TimerID_OVERLAP].elapsed), 
	                  timers->async_begin,currentTime);

      /* Wait on async operation completion */
      ciErrNum = clWaitForEvents(1, (cl_event *)last_event->marker);
      if (ciErrNum != CL_SUCCESS) {
        fprintf(stderr, "Error Waiting for Events!\n");
      }
      pb_Timestamp total_async_time = record_async_times(timers);

      /* Async operations completed before previous CPU operations: 
       * overlapped time is the total async time */
       // If it did finish, then accumulate all the async time that did happen into OVERLAP
       // the immediately preceding EventSynchronize theoretically didn't have any effect since it was already completed.
      if(async_done == CL_COMPLETE /*cudaSuccess*/)
        timers->timers[pb_TimerID_OVERLAP].elapsed += total_async_time;

    } else 
    /* implies (!is_async(timers->current) && asyncs_outstanding(timers)) */
    // i.e. Current Not Async (not KERNEL/COPY_ASYNC) but there are outstanding
    // so something is deeper in stack
    if(async_done == CL_COMPLETE /*cudaSuccess*/) {
      /* Async operations completed before previous CPU operations: 
       * overlapped time is the total async time */
      timers->timers[pb_TimerID_OVERLAP].elapsed += record_async_times(timers);
    }   
    // else, this isn't blocking, so just check the next time around
  }
  
  subtimerlist = timers->sub_timer_list[category];
  struct pb_SubTimer *subtimer = NULL;
  
  if (label != NULL) {
    subtimer = subtimerlist->subtimer_list;
    while (subtimer != NULL) {
      if (strcmp(subtimer->label, label) == 0) {
        break;
      } else {
        subtimer = subtimer->next;
      }
    }
  }

  /* Start the new timer */
  if (category != pb_TimerID_NONE) {
    if(!is_async(category)) {  
      if (subtimerlist != NULL) {
        subtimerlist->current = subtimer;
      }
    
      if (category != timers->current && subtimer != NULL) {
        pb_StartTimerAndSubTimer(&timers->timers[category], &subtimer->timer);
      } else if (subtimer != NULL) {
        pb_StartTimer(&subtimer->timer);
      } else {
        pb_StartTimer(&timers->timers[category]);
      }
    } else {
      if (subtimerlist != NULL) {
        subtimerlist->current = subtimer;
      }
    
      // toSwitchTo Is Async (KERNEL/COPY_ASYNC)
      if (!asyncs_outstanding(timers)) {
        /* No asyncs outstanding, insert a fresh async marker */
        insert_submarker(timers, label, category);
        timers->async_begin = currentTime;
      } else if(!is_async(timers->current)) {
        /* Previous asyncs still in flight, but a previous SwitchTo
         * already marked the end of the most recent async operation, 
         * so we can rename that marker as the beginning of this async 
         * operation */
                  
        struct pb_async_time_marker_list * last_event = get_last_async(timers);
        last_event->timerID = category;
        last_event->label = label;
      } // else, marker for switchToThis was already inserted
      
      //toSwitchto is already asynchronous, but if current/prev state is async too, then DRIVER is already running
      if (!is_async(timers->current)) {
        pb_StartTimer(&timers->timers[pb_TimerID_DRIVER]);
      }
    }
  }
  
  timers->current = category;
#endif
}

void
pb_PrintTimerSet(struct pb_TimerSet *timers)
{
#ifndef DISABLE_PARBOIL_TIMER
  pb_Timestamp wall_end = 0; //get_time();

  struct pb_Timer *t = timers->timers;
  struct pb_SubTimer* sub = NULL;
  
  int maxSubLength;
    
  const char *categories[] = {
    "IO", "Kernel", "Copy", "Driver", "Copy Async", "Compute"
  };
  
  const int maxCategoryLength = 10;
  
  int i;
  for(i = 1; i < pb_TimerID_LAST-1; ++i) { // exclude NONE and OVRELAP from this format
    if(pb_GetElapsedTime(&t[i]) != 0) {
    
      // Print Category Timer
      printf("%-*s: %f\n", maxCategoryLength, categories[i-1], pb_GetElapsedTime(&t[i]));
      
      if (timers->sub_timer_list[i] != NULL) {
        sub = timers->sub_timer_list[i]->subtimer_list;
        maxSubLength = 0;
        while (sub != NULL) {
          // Find longest SubTimer label
          if (strlen(sub->label) > maxSubLength) {
            maxSubLength = strlen(sub->label);
          }
          sub = sub->next;
        }
        
        // Fit to Categories
        if (maxSubLength <= maxCategoryLength) {
         maxSubLength = maxCategoryLength;
        }
        
        sub = timers->sub_timer_list[i]->subtimer_list;
        
        // Print SubTimers
        while (sub != NULL) {
          printf(" -%-*s: %f\n", maxSubLength, sub->label, pb_GetElapsedTime(&sub->timer));
          sub = sub->next;
        }
      }
    }
  }
  
  if(pb_GetElapsedTime(&t[pb_TimerID_OVERLAP]) != 0)
    printf("CPU/Kernel Overlap: %f\n", pb_GetElapsedTime(&t[pb_TimerID_OVERLAP]));
        
  float walltime = (wall_end - timers->wall_begin)/ 1e6;
  printf("Timer Wall Time: %f\n", walltime);
  
#endif
}

void pb_DestroyTimerSet(struct pb_TimerSet * timers)
{
#ifndef DISABLE_PARBOIL_TIMER
  /* clean up all of the async event markers */
  struct pb_async_time_marker_list* event = timers->async_markers;
  while(event != NULL) {

    cl_int ciErrNum = CL_SUCCESS;
    ciErrNum = clWaitForEvents(1, (cl_event *)(event)->marker);
    if (ciErrNum != CL_SUCCESS) {
      //fprintf(stderr, "Error Waiting for Events!\n");
    }
    
    ciErrNum = clReleaseEvent( *((cl_event *)(event)->marker) );
    if (ciErrNum != CL_SUCCESS) {
      fprintf(stderr, "Error Release Events!\n");
    }
    
    free((event)->marker);
    struct pb_async_time_marker_list* next = ((event)->next);

    free(event);

    // (*event) = NULL;
    event = next;
  }

  int i = 0;
  for(i = 0; i < pb_TimerID_LAST; ++i) {    
    if (timers->sub_timer_list[i] != NULL) {
      struct pb_SubTimer *subtimer = timers->sub_timer_list[i]->subtimer_list;
      struct pb_SubTimer *prev = NULL;
      while (subtimer != NULL) {
        free(subtimer->label);
        prev = subtimer;
        subtimer = subtimer->next;
        free(prev);
      }
      free(timers->sub_timer_list[i]);
    }
  }
#endif
}

static pb_Platform** ptr = NULL;

// verbosely print out list of platforms and their devices to the console.
pb_Platform**
pb_GetPlatforms() {
  if (ptr == NULL) {
    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    if (num_platforms == 0) return NULL;

    ptr = (pb_Platform **) malloc(sizeof(pb_Platform *) * (num_platforms + 1));
    cl_platform_id* ids = (cl_platform_id *) malloc(num_platforms * sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, ids, NULL);

    unsigned int i;
    for (i = 0; i < num_platforms; i++) {
      ptr[i] = (pb_Platform *) malloc(sizeof(pb_Platform));
      ptr[i]->clPlatform = ids[i];
      ptr[i]->contexts = NULL;
      ptr[i]->in_use = 0;
      ptr[i]->devices = NULL;

      size_t sz;
      clGetPlatformInfo(ids[i], CL_PLATFORM_NAME, 0, NULL, &sz);
      char* name = (char *) malloc(sz + 1);
      clGetPlatformInfo(ids[i], CL_PLATFORM_NAME, sz, name, NULL);
      name[sz] = '\0';
      ptr[i]->name = name;

      clGetPlatformInfo(ids[i], CL_PLATFORM_VERSION, 0, NULL, &sz);
      char* version = (char *) malloc(sz + 1);
      clGetPlatformInfo(ids[i], CL_PLATFORM_VERSION, sz, version, NULL);
      version[sz] = '\0';
      ptr[i]->version = version;
    }
    ptr[i] = NULL;

    free(ids);
  }

  return (pb_Platform**) ptr;
}

pb_Context* 
createContext(pb_Platform* pb_platform, pb_Device* pb_device) {
  pb_Context* c = (pb_Context*) malloc(sizeof(pb_Context));
  cl_int clStatus;
  cl_context_properties clCps[3] = {
    CL_CONTEXT_PLATFORM, (cl_context_properties)(pb_platform->clPlatform), 0
  };
  c->clContext =
    clCreateContext(clCps, 1, (cl_device_id*)&pb_device->clDevice, NULL, NULL, &clStatus);
  c->clPlatformId = pb_platform->clPlatform;
  c->clDeviceId = pb_device->clDevice;
  c->pb_platform = pb_platform;
  c->pb_device = pb_device;
  pb_platform->in_use = 1;
  pb_device->in_use = 1;
  unsigned int i = 0;
  if (pb_platform->contexts == NULL) {
    pb_platform->contexts = (pb_Context**) malloc(2*sizeof(pb_Context*));
  } else {
    for (i = 0; pb_platform->contexts[i] != NULL; i++) {};
    pb_platform->contexts = (pb_Context**) realloc(pb_platform->contexts,
                                                   (i+1)*sizeof(pb_Context*));
  }
  pb_platform->contexts[i+1] = NULL;
  pb_platform->contexts[i] = c;
  return c;
}

// choose a platform by name.
pb_Platform*
pb_GetPlatformByName(const char* name) {
  pb_Platform** ps = (pb_Platform **) pb_GetPlatforms();
  if (ps == NULL) return NULL;
  if (name == NULL) {
    return *ps;
  }

  while (*ps) {
    if (strstr((*ps)->name, name)) break;
    ps++;
  }
  return (pb_Platform*) *ps;
}

pb_Device**
pb_GetDevices(pb_Platform* pb_platform) {
  if (pb_platform->devices == NULL) {
    cl_uint num_devs;
    cl_device_id* dev_ids;
    clGetDeviceIDs((cl_platform_id) pb_platform->clPlatform,
                    CL_DEVICE_TYPE_ALL, 0, NULL, &num_devs);
    if (num_devs == 0) return NULL;

    pb_platform->devices =
      (pb_Device **) malloc((num_devs + 1) * sizeof(pb_Device *));
    dev_ids = (cl_device_id *) malloc(sizeof(cl_device_id) * num_devs);
    clGetDeviceIDs((cl_platform_id) pb_platform->clPlatform,
                    CL_DEVICE_TYPE_ALL, num_devs, dev_ids, NULL);

    unsigned int i;
    for (i = 0; i < num_devs; i++) {
      pb_platform->devices[i] = (pb_Device *) malloc(sizeof(pb_Device));

      pb_platform->devices[i]->clDevice = dev_ids[i];
      pb_platform->devices[i]->id = i;

      size_t sz;
      clGetDeviceInfo(dev_ids[i], CL_DEVICE_NAME, 0, NULL, &sz);
      char* name = (char *) malloc(sz + 1);
      clGetDeviceInfo(dev_ids[i], CL_DEVICE_NAME, sz, name, NULL);
      name[sz] = '\0';
      pb_platform->devices[i]->name = (char *) name;

      cl_bool available;
      clGetDeviceInfo(dev_ids[i], CL_DEVICE_AVAILABLE, sizeof(cl_bool), &available, NULL); 
      pb_platform->devices[i]->available = (int) available;

      pb_platform->devices[i]->in_use = 0;
    }
    pb_platform->devices[i] = NULL;
  }
  return (pb_Device **) pb_platform->devices;
}

// choose a device by name.
static pb_Device*
pb_SelectDeviceByName(pb_Device **ds, const char* name) {
  if (ds == NULL) return NULL;
  if (name == NULL) return *ds;
  while (*ds) {
    if (strstr((*ds)->name, name)) break;
    ds++;
  }

  return *ds;
}

// choose a device by name and set the device's 'in_use' flag.
pb_Device*
pb_GetDeviceByName(pb_Platform* pb_platform, const char* name) {
  pb_Device** ds = (pb_Device **) pb_GetDevices(pb_platform);
  pb_Device *d = pb_SelectDeviceByName(ds, name);

  if (d) d->in_use = 1;

  return d;
}

void
pb_ReleasePlatforms() {
  if (!ptr) return;
  pb_Platform** cur_ptr = ptr;
  while (*cur_ptr) {
    pb_Platform* pfptr = *cur_ptr++;
    if (pfptr->devices) {
      pb_Device** dvptr = pfptr->devices;
      while (*dvptr) {
        pb_Device* d = *dvptr++;
        free(d->name);
        free(d);
      }
      free(pfptr->devices);
    }
    if (pfptr->contexts) {
      pb_Context** cptr = pfptr->contexts;
      while (*cptr) {
        free(*cptr++);
      }
      free(pfptr->contexts);
    }
    free(pfptr->name);
    free(pfptr);
  }
  free(ptr);
  ptr = NULL;
}

pb_Platform*
pb_GetPlatformByNameAndVersion(const char* name, const char* version) {
  pb_Platform** ps = (pb_Platform **) pb_GetPlatforms();
  if (ps == NULL) return NULL;
  if (name == NULL) return *ps;
  while (*ps) {
    if (strstr((*ps)->name, name) && strstr((*ps)->version, version)) break;
    ps++;
  }
  return (pb_Platform*) *ps;
}

/* Return a pointer to the device at the specified index, or NULL.
 * Used by pb_GetDevice. */
static pb_Device *
select_device_by_index(pb_Device** ds, int id)
{
  int i = 0;
  pb_Device** p = ds;
  while (*p && (i < id)) { p++; i++; }
  return *p;
}

/* Return a pointer to the device with the specified type, or NULL.
 * Used by pb_GetDevice. */
static pb_Device *
select_device_by_type(pb_Device** ds,
                      enum pb_DeviceSelectionCriterion criterion)
{
  cl_device_type sought_type;

  /* Determine the OpenCL device type to search for */
  switch(criterion) {
  case pb_Device_CPU:
    sought_type = CL_DEVICE_TYPE_CPU;
    break;
  case pb_Device_GPU:
    sought_type = CL_DEVICE_TYPE_GPU;
    break;
  case pb_Device_ACCELERATOR:
    sought_type = CL_DEVICE_TYPE_ACCELERATOR;
    break;
  default:
    fprintf(stderr, "pb_GetDevice: Invalid device type");
    exit(-1);
  }

  /* Find the device */
  {
    pb_Device** p = ds;
    cl_device_type type;
    while (*p) {
      clGetDeviceInfo(((cl_device_id) ((*p)->clDevice)), CL_DEVICE_TYPE,
                      sizeof(cl_device_type), &type, NULL);
      if (type == sought_type) break;
    }

    return *p;
  }
}

pb_Device*
pb_GetDevice(pb_Platform* pb_platform, struct pb_DeviceParam *device)
{
  pb_Device** ds = (pb_Device **) pb_GetDevices(pb_platform);

  // The list of devices must be nonempty
  if (ds == NULL || *ds == NULL) {
    fprintf(stderr, "Error: No device is found in platform: name = %s, version = %s\n.", pb_platform->name, pb_platform->version);
    exit(-1);
  }

  pb_Device *selected_device = NULL;

  if (device != NULL) {
    /* Use 'device' to select and return a device.
     * If unable to select a device, fall
     * back on the default selection mechanism. */
    switch(device->criterion) {
    case pb_Device_INDEX:
      selected_device = select_device_by_index(ds, device->index);
      break;
    case pb_Device_GPU:
    case pb_Device_CPU:
    case pb_Device_ACCELERATOR:
      selected_device = select_device_by_type(ds, device->criterion);
      break;
    case pb_Device_NAME:
      selected_device = pb_SelectDeviceByName(ds, device->name);
      break;
    default:
      fprintf(stderr, "pb_GetDevice: Invalid argument");
      exit(-1);
    }
  }

  /* By default or if user-specified selection failed,
   * select the first device */
  if (selected_device == NULL)
    selected_device = *ds;

  /* Set the in_use flag */
  selected_device->in_use = 1;

  return selected_device;
}

pb_Device*
pb_GetDeviceByEnvVars(pb_Platform* pb_platform) {

  /* Convert environment variables to a 'pb_DeviceParam' */
  struct pb_DeviceParam *param = NULL;

  char* device_num = getenv("PARBOIL_DEVICE_NUMBER");
  if (device_num && strcmp(device_num, "")) {
    int id = atoi(device_num);
    param = pb_DeviceParam_index(id);
  }
  else {
    char* device_name = getenv("PARBOIL_DEVICE_NAME");
    if (device_name && strcmp(device_name, "")) {
      param = pb_DeviceParam_name(strdup(device_name));
    }
    else {
      char* device_type = getenv("PARBOIL_DEVICE_TYPE");
      if (device_type && strcmp(device_type, "")) {
        if (strcmp(device_type, "CPU") == 0)
          param = pb_DeviceParam_cpu();
        else if (strcmp(device_type, "GPU") == 0)
          param = pb_DeviceParam_gpu();
        else if (strcmp(device_type, "ACCELERATOR") == 0)
          param = pb_DeviceParam_accelerator();
      }
    }
  }

  /* Get a device */
  pb_Device *d = pb_GetDevice(pb_platform, param);
  pb_FreeDeviceParam(param);

  return d;
}

pb_Platform*
pb_GetPlatformByEnvVars() {
  char* name = getenv("PARBOIL_PLATFORM_NAME");
  char* version = getenv("PARBOIL_PLATFORM_VERSION");

  /* Create a pb_PlatformParam object (or NULL) representing the data from the
   * environment variables */
  struct pb_PlatformParam *platform;

  if (name) {
    if (version) {
      platform = pb_PlatformParam(strdup(name), strdup(version));
    }
    else {
      platform = pb_PlatformParam(strdup(name), NULL);
    }
  }
  else {
    platform = NULL;
  }

  /* Convert to a platform */
  pb_Platform *p = pb_GetPlatform(platform);
  pb_FreePlatformParam(platform);

  return p;
}

/* Choose an OpenCL platform based on the given command-line parameters.
 * If NULL, use the default OpenCL platform. */
pb_Platform*
pb_GetPlatform(struct pb_PlatformParam *platform) {
  if (platform != NULL) {
    /* Try to use command-line parameters to choose platform */
    char *name = platform->name;
    char *version = platform->version;

    if (!name) {
      fprintf(stderr, "Internal error: NULL pointer");
      exit(-1);
    }

    if (version) {
      pb_Platform* p = pb_GetPlatformByNameAndVersion(name, version);
      if (p) return p;
    }

    pb_Platform* p = pb_GetPlatformByName(name);
    if (p) return p;
  }

  pb_Platform* p = pb_GetPlatformByName(NULL);
  if (p == NULL) {
    fprintf(stderr, "Error: No OpenCL platform in this system. Exiting.");
    exit(-1);
  }
  return p;
}

//extern void perf_init();
//extern void mxpa_scheduler_init();

pb_Context*
pb_InitOpenCLContext(struct pb_Parameters* parameters) {
#if 0
  pb_Platform* ps = pb_GetPlatform(parameters->platform);
  if (!ps) return NULL;
  pb_Device* ds = pb_GetDevice(ps, parameters->device);
  if (!ds) return NULL;

  /* HERE INITIALIZE TIMER */
  //perf_init();
  //mxpa_scheduler_init();

  pb_Context* c = createContext(ps, ds);
  pb_PrintPlatformInfo(c);
  return c;
#endif
  cl_int _err;
  cl_platform_id platform_id;
  cl_device_id device_id;
  cl_context context;
  clGetPlatformIDs(1, &platform_id, NULL);
  clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
  context = clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err);

  pb_Context* c = (pb_Context*)malloc(sizeof(pb_Context));
  c->clContext = context;
  c->clDeviceId = device_id;
  c->clPlatformId = platform_id;
  c->pb_platform = (pb_Platform*)malloc(sizeof(pb_Platform));
  c->pb_device = (pb_Device*)malloc(sizeof(pb_Device));  
  c->pb_platform->devices = (pb_Device**)malloc(sizeof(pb_Device*) * 2);  
  c->pb_platform->devices[0] = c->pb_device;
  c->pb_platform->devices[1] = NULL;
  c->pb_platform->contexts = (pb_Context**)malloc(sizeof(pb_Context*) * 2);  
  c->pb_platform->contexts[0] = c;
  c->pb_platform->contexts[1] = NULL;
  c->pb_platform->in_use = 1;
  c->pb_device->in_use = 1;  
  return c;
}

void
pb_ReleaseOpenCLContext(pb_Context* c) {
  pb_ReleasePlatforms();
}

void
pb_PrintPlatformInfo(pb_Context* c) {
  /*pb_Platform** ps = pb_GetPlatforms();
  if (!ps) {
    fprintf (stderr, "No platform found");
    return;
  }

  printf ("********************************************************\n");
  printf ("DETECTED OPENCL PLATFORMS AND DEVICES:\n");
  printf ("--------------------------------------------------------\n");

  while (*ps) {
    printf ("PLATFORM = %s, %s", (*ps)->name, (*ps)->version); 
    if (c->pb_platform == *ps) printf (" (SELECTED)");
    printf ("\n");

    pb_Device** ds = (pb_Device **) pb_GetDevices((*ps));
    if (ds == NULL) {
      printf ("  + (No devices)\n");
    } else {
      while (*ds) {
        printf ("  + %d: %s", (*ds)->id, (*ds)->name);
        if (c->pb_device == *ds) printf (" (SELECTED)");
        printf ("\n");
        ds++;
      }
    }

    ps++;
  }
  printf ("********************************************************\n");*/
}

#ifdef MEASURE_KERNEL_TIME

#undef clEnqueueNDRangeKernel

//extern void pin_trace_enable(char*);
//extern void pin_trace_disable(char*);

cl_int
pb_clEnqueueNDRangeKernel(cl_command_queue  q/* command_queue */,
                       cl_kernel            k/* kernel */,
                       cl_uint              d/* work_dim */,
                       const size_t *       o/* global_work_offset */,
                       const size_t *       gws/* global_work_size */,
                       const size_t *       lws/* local_work_size */,
                       cl_uint              n/* num_events_in_wait_list */,
                       const cl_event *     w/* event_wait_list */,
                       cl_event *           e/* event */) {

  char buf[128];
  struct timeval begin, end;
  clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, 128, buf, NULL);

#if 0
  int i;
  for (i = 0; i  < d; i++) {
    printf ("%s: %d: %d / %d\n", buf, i, gws[i], (lws == NULL ? 0 : lws[i]));
  }
#endif

  clFinish(q); clFlush(q);
  //pin_trace_enable(buf);
  //gettimeofday(&begin, NULL);
  cl_int result = clEnqueueNDRangeKernel(q, k, d, o, gws, lws, n, w, e);
  clFinish(q); clFlush(q);
  //gettimeofday(&end, NULL);
  //pin_trace_disable(buf);
  //float t = (float)(end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / 1000000.0f;
  fflush(stdout);
  fflush(stderr);
  //printf ("PBTIMER: %s: %f\n", buf, t);
  return result;
}

#endif

void
pb_sig_float(char* c, float* p, int sz) {
  int i;
  double s = 0.0;
  for (i = 0; i < sz; i++) s += p[i] * (float)(i+1);
  printf ("[Signature] %s = %lf\n", c, s);
}

void
pb_sig_double(char* c, double* p, int sz) {
  int i;
  double s = 0.0;
  for (i = 0; i < sz; i++) s += p[i];
  printf ("[Signature] %s = %lf\n", c, s);
}

void
pb_sig_short(char* c, short* p, int sz) {
  int i;
  long long int s = 0;
  for (i = 0; i < sz; i++) s += p[i];
  printf ("[Signature] %s = %lld\n", c, s);
}

void
pb_sig_int(char* c, int* p, int sz) {
  int i;
  long long int s = 0;
  for (i = 0; i < sz; i++) s += p[i];
  printf ("[Signature] %s = %lld\n", c, s);
}

void
pb_sig_uchar(char* c, unsigned char* p, unsigned int sz) {
  int i;
  unsigned long long int s = 0;
  for (i = 0; i < sz; i++) s += p[i];
  printf ("[Signature] %s = %lld\n", c, s);
}

void pb_sig_clmem(char* s, cl_command_queue command_queue, cl_mem memobj, int ty) {
  size_t sz;
  if (clGetMemObjectInfo(memobj, CL_MEM_SIZE, sizeof(size_t), &sz, NULL) != CL_SUCCESS) {
    printf ("Something wrong.\n");
    assert(0);
  } else {
    printf ("size = %ld\n", sz);
  }
  char* hp; // = (char*) malloc(sz);
  //posix_memalign((void**)&hp, 64, sz);
  hp = (char*)malloc(sz);

  clEnqueueReadBuffer (command_queue,
  memobj,
  CL_TRUE,
  0,
  sz,
  hp,
  0,
  NULL,
  NULL);

  if (ty == T_FLOAT) pb_sig_float(s, (float*)hp, sz/sizeof(float));
  if (ty == T_DOUBLE) pb_sig_double(s, (double*)hp, sz/sizeof(double));
  if (ty == T_INT) pb_sig_int(s, (int*)hp, sz/sizeof(int));
  if (ty == T_SHORT) pb_sig_short(s, (short*)hp, sz/sizeof(short));
  if (ty == T_UCHAR) pb_sig_uchar(s, (unsigned char*)hp, sz/sizeof(char));

  free(hp);
}

