#include <sys/stat.h>
#include <newlib.h>
#include <unistd.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <string.h>
 
int _close(int file) { return -1; }
 
int _fstat(int file, struct stat *st) { return -1; }
 
int _isatty(int file) { return 0; }
 
int _lseek(int file, int ptr, int dir) { return 0; }
 
int _open(const char *name, int flags, int mode) { return -1; }
 
int _read(int file, char *ptr, int len) { return -1; }
 
caddr_t _sbrk(int incr) { 
  __asm__ __volatile__("ebreak");
  return 0; 
}
 
int _write(int file, char *ptr, int len) {
  int i; 
  for (i = 0; i < len; ++i) {
    vx_putchar(*ptr++);
  }
  return len;
}

int _kill(int pid, int sig) { return -1; }

int _getpid() {
  return vx_warp_gid();
}

void __init_tls(void) {  
  extern char __tdata_start[];
  extern char __tbss_offset[];
  extern char __tdata_size[];
  extern char __tbss_size[];

  // activate all threads
  vx_tmc(-1);

  // TLS memory initialization
  register char *__thread_self __asm__ ("tp");
  memcpy(__thread_self, __tdata_start, (size_t)__tdata_size);
  memset(__thread_self + (size_t)__tbss_offset, 0, (size_t)__tbss_size);

  // back to single thread execution
  vx_tmc(0 == vx_warp_id());
}

 #ifdef HAVE_INITFINI_ARRAY

/* These magic symbols are provided by the linker.  */
extern void (*__preinit_array_start []) (void) __attribute__((weak));
extern void (*__preinit_array_end []) (void) __attribute__((weak));
extern void (*__init_array_start []) (void) __attribute__((weak));
extern void (*__init_array_end []) (void) __attribute__((weak));

#ifdef HAVE_INIT_FINI
extern void _init (void);
#endif

/* Iterate over all the init routines.  */
void __libc_init_array (void) {
  size_t count;
  size_t i;

  count = __preinit_array_end - __preinit_array_start;
  for (i = 0; i < count; i++)
    __preinit_array_start[i] ();

#ifdef HAVE_INIT_FINI
  _init ();
#endif

  count = __init_array_end - __init_array_start;
  for (i = 0; i < count; i++)
    __init_array_start[i] ();
}
#endif

#ifdef HAVE_INITFINI_ARRAY
extern void (*__fini_array_start []) (void) __attribute__((weak));
extern void (*__fini_array_end []) (void) __attribute__((weak));

#ifdef HAVE_INIT_FINI
extern void _fini (void);
#endif

/* Run all the cleanup routines.  */
void __libc_fini_array (void) {
  size_t count;
  size_t i;
  
  count = __fini_array_end - __fini_array_start;
  for (i = count; i > 0; i--)
    __fini_array_start[i-1] ();

#ifdef HAVE_INIT_FINI
  _fini ();
#endif
}
#endif