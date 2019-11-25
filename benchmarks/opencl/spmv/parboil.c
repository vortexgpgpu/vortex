/*
 * (c) 2007 The Board of Trustees of the University of Illinois.
 */

#include <parboil.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if _POSIX_VERSION >= 200112L
# include <sys/time.h>
#endif


/*****************************************************************************/
/* Timer routines */

static void
accumulate_time(pb_Timestamp *accum,
		pb_Timestamp start,
		pb_Timestamp end)
{
#if _POSIX_VERSION >= 200112L
  *accum += end - start;
#else
# error "Timestamps not implemented for this system"
#endif
}

#if _POSIX_VERSION >= 200112L
static pb_Timestamp get_time()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (pb_Timestamp) (tv.tv_sec * 1000000LL + tv.tv_usec);
}
#else
# error "no supported time libraries are available on this platform"
#endif

void
pb_ResetTimer(struct pb_Timer *timer)
{
  timer->state = pb_Timer_STOPPED;

#if _POSIX_VERSION >= 200112L
  timer->elapsed = 0;
#else
# error "pb_ResetTimer: not implemented for this system"
#endif
}

void
pb_StartTimer(struct pb_Timer *timer)
{
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
}

void
pb_StartTimerAndSubTimer(struct pb_Timer *timer, struct pb_Timer *subtimer)
{
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

}

void
pb_StopTimer(struct pb_Timer *timer)
{

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

}

void pb_StopTimerAndSubTimer(struct pb_Timer *timer, struct pb_Timer *subtimer) {

  pb_Timestamp fini;

  unsigned int numNotRunning = 0x3; // 0b11
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

}

/* Get the elapsed time in seconds. */
double
pb_GetElapsedTime(struct pb_Timer *timer)
{
  double ret;

  if (timer->state != pb_Timer_STOPPED) {
    fputs("Elapsed time from a running timer is inaccurate\n", stderr);
  }

#if _POSIX_VERSION >= 200112L
  ret = timer->elapsed / 1e6;
#else
# error "pb_GetElapsedTime: not implemented for this system"
#endif
  return ret;
}

void
pb_InitializeTimerSet(struct pb_TimerSet *timers)
{
  int n;
  
  timers->wall_begin = get_time();

  timers->current = pb_TimerID_NONE;

  timers->async_markers = NULL;
  

  for (n = 0; n < pb_TimerID_LAST; n++) {
    pb_ResetTimer(&timers->timers[n]);
    timers->sub_timer_list[n] = NULL; // free first?
  }
}

void
pb_AddSubTimer(struct pb_TimerSet *timers, char *label, enum pb_TimerID pb_Category) {  
  
  struct pb_SubTimer *subtimer = (struct pb_SubTimer *) malloc
    (sizeof(struct pb_SubTimer));
    
  int len = strlen(label);
    
  subtimer->label = (char *) malloc (sizeof(char)*(len+1));
  sprintf(subtimer->label, "%s\0", label);
  
  pb_ResetTimer(&subtimer->timer);
  subtimer->next = NULL;
  
  struct pb_SubTimerList *subtimerlist = timers->sub_timer_list[pb_Category];
  if (subtimerlist == NULL) {
    subtimerlist = (struct pb_SubTimerList *) malloc
      (sizeof(struct pb_SubTimerList));
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
  
}

void
pb_SwitchToSubTimer(struct pb_TimerSet *timers, char *label, enum pb_TimerID category)
{

// switchToSub( NULL, NONE
// switchToSub( NULL, some
// switchToSub( some, some
// switchToSub( some, NONE -- tries to find "some" in NONE's sublist, which won't be printed
  
  struct pb_Timer *topLevelToStop = NULL;
  if (timers->current != category && timers->current != pb_TimerID_NONE) {
    // Switching to subtimer in a different category needs to stop the top-level current, different categoried timer.
    // NONE shouldn't have a timer associated with it, so exclude from branch
    topLevelToStop = &timers->timers[timers->current];
  } 

  struct pb_SubTimerList *subtimerlist = timers->sub_timer_list[timers->current];
  struct pb_SubTimer *curr = (subtimerlist == NULL) ? NULL : subtimerlist->current;
  
  if (timers->current != pb_TimerID_NONE) {
    if (curr != NULL && topLevelToStop != NULL) {
      pb_StopTimerAndSubTimer(topLevelToStop, &curr->timer);
    } else if (curr != NULL) {
      pb_StopTimer(&curr->timer);
    } else {
      pb_StopTimer(topLevelToStop);
    }
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
  
  if (category != pb_TimerID_NONE) {
    
    if (subtimerlist != NULL) {
      subtimerlist->current = subtimer;
    }
    
    if (category != timers->current && subtimer != NULL) {
      pb_StartTimerAndSubTimer(&timers->timers[category], &subtimer->timer);
    } else if (subtimer != NULL) {
      // Same category, different non-NULL subtimer
      pb_StartTimer(&subtimer->timer);
    } else{
      // Different category, but no subtimer (not found or specified as NULL) -- unprefered way of setting topLevel timer
      pb_StartTimer(&timers->timers[category]);
    }
  }  
  
  timers->current = category;
  
}

void
pb_SwitchToTimer(struct pb_TimerSet *timers, enum pb_TimerID timer)
{
  /* Stop the currently running timer */
  /*if (timers->current != pb_TimerID_NONE) {
    struct pb_SubTimer *currSubTimer = NULL;
    struct pb_SubTimerList *subtimerlist = timers->sub_timer_list[timers->current];
    
    if ( subtimerlist != NULL) {
      currSubTimer = timers->sub_timer_list[timers->current]->current;
    }
    if ( currSubTimer!= NULL) {
      pb_StopTimerAndSubTimer(&timers->timers[timers->current], &currSubTimer->timer);
    } else {
      pb_StopTimer(&timers->timers[timers->current]);
    }    
  }

  timers->current = timer;

  if (timer != pb_TimerID_NONE) {
    pb_StartTimer(&timers->timers[timer]);
  }*/
}

void
pb_PrintTimerSet(struct pb_TimerSet *timers)
{

  pb_Timestamp wall_end = get_time();

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
  
}

void pb_DestroyTimerSet(struct pb_TimerSet * timers)
{
  /* clean up all of the async event markers */
  struct pb_async_time_marker_list ** event = &(timers->async_markers);
  while( *event != NULL) {
    struct pb_async_time_marker_list ** next = &((*event)->next);
    free(*event);
    (*event) = NULL;
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
}


