/*
 * (c) 2010 The Board of Trustees of the University of Illinois.
 */
#ifndef PARBOIL_HEADER
#define PARBOIL_HEADER

#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <unistd.h>

/* A platform as specified by the user on the command line */
struct pb_PlatformParam {
  char *name;                   /* The platform name.  This string is owned. */
  char *version;                /* The platform version; may be NULL.
                                 * This string is owned. */
};

/* Create a PlatformParam from the given strings.
 * 'name' must not be NULL.  'version' may be NULL.
 * If not NULL, the strings should have been allocated by malloc(),
 * and they will be owned by the returned object.
 */
struct pb_PlatformParam *
pb_PlatformParam(char *name, char *version);

void
pb_FreePlatformParam(struct pb_PlatformParam *);

/* A criterion for how to select a device */
enum pb_DeviceSelectionCriterion {
  pb_Device_INDEX,             /* Enumerate the devices and select one
                                * by its number */
  pb_Device_CPU,                /* Select a CPU device */
  pb_Device_GPU,                /* Select a GPU device  */
  pb_Device_ACCELERATOR,        /* Select an accelerator device */
  pb_Device_NAME                /* Select a device by name */
};

/* A device as specified by the user on the command line */
struct pb_DeviceParam {
  enum pb_DeviceSelectionCriterion criterion;
  union {
    int index;                  /* If criterion == pb_Device_INDEX,
                                 * the index of the device */
    char *name;                 /* If criterion == pb_Device_NAME,
                                 * the name of the device.
                                 * This string is owned. */
  };
};

struct pb_DeviceParam *
pb_DeviceParam_index(int index);

struct pb_DeviceParam *
pb_DeviceParam_cpu(void);

struct pb_DeviceParam *
pb_DeviceParam_gpu(void);

struct pb_DeviceParam *
pb_DeviceParam_accelerator(void);

/* Create a by-name device selection criterion.
 * The string should have been allocated by malloc(), and it will will be
 * owned by the returned object.
 */
struct pb_DeviceParam *
pb_DeviceParam_name(char *name);

void
pb_FreeDeviceParam(struct pb_DeviceParam *);

/* Command line parameters for benchmarks */
struct pb_Parameters {
  char *outFile;		/* If not NULL, the raw output of the
				 * computation should be saved to this
				 * file. The string is owned. */
  char **inpFiles;		/* A NULL-terminated array of strings
				 * holding the input file(s) for the
				 * computation.  The array and strings
				 * are owned. */
  struct pb_PlatformParam *platform; /* If not NULL, the platform
                                      * specified on the command line. */
  struct pb_DeviceParam *device; /* If not NULL, the device
                                      * specified on the command line. */
};

/* Read command-line parameters.
 *
 * The argc and argv parameters to main are read, and any parameters
 * interpreted by this function are removed from the argument list.
 *
 * A new instance of struct pb_Parameters is returned.
 * If there is an error, then an error message is printed on stderr
 * and NULL is returned.
 */
struct pb_Parameters *
pb_ReadParameters(int *_argc, char **argv);

/* Free an instance of struct pb_Parameters.
 */
void
pb_FreeParameters(struct pb_Parameters *p);

void
pb_FreeStringArray(char **);

/* Count the number of input files in a pb_Parameters instance.
 */
int
pb_Parameters_CountInputs(struct pb_Parameters *p);

/* A time or duration. */
//#if _POSIX_VERSION >= 200112L
typedef unsigned long long pb_Timestamp; /* time in microseconds */
//#else
//# error "Timestamps not implemented"
//#endif

enum pb_TimerState {
  pb_Timer_STOPPED,
  pb_Timer_RUNNING,
};

struct pb_Timer {
  enum pb_TimerState state;
  pb_Timestamp elapsed;		/* Amount of time elapsed so far */
  pb_Timestamp init;		/* Beginning of the current time interval,
				 * if state is RUNNING.  End of the last 
				 * recorded time interfal otherwise.  */
};

/* Reset a timer.
 * Use this to initialize a timer or to clear
 * its elapsed time.  The reset timer is stopped.
 */
void
pb_ResetTimer(struct pb_Timer *timer);

/* Start a timer.  The timer is set to RUNNING mode and
 * time elapsed while the timer is running is added to
 * the timer.
 * The timer should not already be running.
 */
void
pb_StartTimer(struct pb_Timer *timer);

/* Stop a timer.
 * This stops adding elapsed time to the timer.
 * The timer should not already be stopped.
 */
void
pb_StopTimer(struct pb_Timer *timer);

/* Get the elapsed time in seconds. */
double
pb_GetElapsedTime(struct pb_Timer *timer);

/* Execution time is assigned to one of these categories. */
enum pb_TimerID {
  pb_TimerID_NONE = 0,
  pb_TimerID_IO,		/* Time spent in input/output */
  pb_TimerID_KERNEL,		/* Time spent computing on the device, 
				 * recorded asynchronously */
  pb_TimerID_COPY,		/* Time spent synchronously moving data 
				 * to/from device and allocating/freeing 
				 * memory on the device */
  pb_TimerID_DRIVER,		/* Time spent in the host interacting with the 
				 * driver, primarily for recording the time 
                                 * spent queueing asynchronous operations */
  pb_TimerID_COPY_ASYNC,	/* Time spent in asynchronous transfers */
  pb_TimerID_COMPUTE,		/* Time for all program execution other
				 * than parsing command line arguments,
				 * I/O, kernel, and copy */
  pb_TimerID_OVERLAP,		/* Time double-counted in asynchronous and 
				 * host activity: automatically filled in, 
				 * not intended for direct usage */
  pb_TimerID_LAST		/* Number of timer IDs */
};

/* Dynamic list of asynchronously tracked times between events */
struct pb_async_time_marker_list {
  char *label; // actually just a pointer to a string
  enum pb_TimerID timerID;	/* The ID to which the interval beginning 
                                 * with this marker should be attributed */
  void * marker; 
  //cudaEvent_t marker; 		/* The driver event for this marker */
  struct pb_async_time_marker_list *next; 
};

struct pb_SubTimer {
  char *label;
  struct pb_Timer timer;
  struct pb_SubTimer *next;
};

struct pb_SubTimerList {
  struct pb_SubTimer *current;
  struct pb_SubTimer *subtimer_list;
};

/* A set of timers for recording execution times. */
struct pb_TimerSet {
  enum pb_TimerID current;
  struct pb_async_time_marker_list* async_markers;
  pb_Timestamp async_begin;
  pb_Timestamp wall_begin;
  struct pb_Timer timers[pb_TimerID_LAST];
  struct pb_SubTimerList *sub_timer_list[pb_TimerID_LAST];
};

/* Reset all timers in the set. */
void
pb_InitializeTimerSet(struct pb_TimerSet *timers);

void
pb_AddSubTimer(struct pb_TimerSet *timers, char *label, enum pb_TimerID pb_Category);

/* Select which timer the next interval of time should be accounted
 * to. The selected timer is started and other timers are stopped.
 * Using pb_TimerID_NONE stops all timers. */
void
pb_SwitchToTimer(struct pb_TimerSet *timers, enum pb_TimerID timer);

void
pb_SwitchToSubTimer(struct pb_TimerSet *timers, char *label, enum pb_TimerID category);

/* Print timer values to standard output. */
void
pb_PrintTimerSet(struct pb_TimerSet *timers);

/* Release timer resources */
void
pb_DestroyTimerSet(struct pb_TimerSet * timers);

void
pb_SetOpenCL(void *clContextPtr, void *clCommandQueuePtr);


typedef struct pb_Device_tag {
  char* name;
  void* clDevice;
  int id;
  unsigned int in_use;
  unsigned int available;
} pb_Device;

struct pb_Context_tag;
typedef struct pb_Context_tag pb_Context;

typedef struct pb_Platform_tag {
  char* name;
  char* version;
  void* clPlatform;
  unsigned int in_use;
  pb_Context** contexts;
  pb_Device** devices;
} pb_Platform;

struct pb_Context_tag {
  void* clPlatformId;
  void* clContext;
  void* clDeviceId;
  pb_Platform* pb_platform;
  pb_Device* pb_device;
};

// verbosely print out list of platforms and their devices to the console.
pb_Platform**
pb_GetPlatforms();

// Choose a platform according to the given platform specification
pb_Platform*
pb_GetPlatform(struct pb_PlatformParam *platform);

// choose a platform: by name, name & version
pb_Platform*
pb_GetPlatformByName(const char* name);

pb_Platform*
pb_GetPlatformByNameAndVersion(const char* name, const char* version);

// Choose a device according to the given device specification
pb_Device*
pb_GetDevice(pb_Platform* pb_platform, struct pb_DeviceParam *device);

pb_Device**
pb_GetDevices(pb_Platform* pb_platform);

// choose a device by name.
pb_Device*
pb_GetDeviceByName(pb_Platform* pb_platform, const char* name);

pb_Platform*
pb_GetPlatformByEnvVars();

pb_Context*
pb_InitOpenCLContext(struct pb_Parameters* parameters);

void
pb_ReleasePlatforms();

void
pb_ReleaseContext(pb_Context* c);

void
pb_PrintPlatformInfo(pb_Context* c);

void
perf_init();

//#define MEASURE_KERNEL_TIME

#include <CL/cl.h>

#ifdef MEASURE_KERNEL_TIME
#define clEnqueueNDRangeKernel(q,k,d,o,dg,db,a,b,c) pb_clEnqueueNDRangeKernel((q), (k), (d), (o), (dg), (db), (a), (b), (c))
cl_int
pb_clEnqueueNDRangeKernel(cl_command_queue /* command_queue */,
                       cl_kernel        /* kernel */,
                       cl_uint          /* work_dim */,
                       const size_t *   /* global_work_offset */,
                       const size_t *   /* global_work_size */,
                       const size_t *   /* local_work_size */,
                       cl_uint          /* num_events_in_wait_list */,
                       const cl_event * /* event_wait_list */,
                       cl_event *       /* event */);
#endif

enum { T_FLOAT, T_DOUBLE, T_SHORT, T_INT, T_UCHAR };
void pb_sig_float(char*, float*, int);
void pb_sig_double(char*, double*, int);
void pb_sig_short(char*, short*, int);
void pb_sig_int(char*, int*, int);
void pb_sig_uchar(char*, unsigned char*, unsigned int);
void pb_sig_clmem(char*, cl_command_queue, cl_mem, int);

#ifdef __cplusplus
}
#endif

#endif //PARBOIL_HEADER

