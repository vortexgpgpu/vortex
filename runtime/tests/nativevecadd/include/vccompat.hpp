/* pocl/include/vccompat.h - Compatibility header to provide some functions 
   which are not found from VC++. 

   All functions should be static inline so that they can be included in many places
   without having problem of symbol collision.

   Copyright (c) 2014 Mikael Lepistö <elhigu@gmail.com>
   
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#ifndef VCCOMPAT_HPP
#define VCCOMPAT_HPP

#include <Windows.h>
#define __restrict__ __restrict
#define restrict __restrict

#include <intrin.h>
#define __builtin_popcount __popcnt

// ERROR is used as label for goto in some OCL API functions
#undef ERROR

// if this causes linking problems, use inline function below...
#define snprintf _snprintf

/*
static inline int snprintf(char *str, size_t size, const char *format, ...) {
  va_list args;
  va_start(args, format);
  _snprintf(str, size, format, args);
  va_end(args);
}
*/

static inline char* strtok_r(char *str, const char *delim, char **saveptr) {
  return strtok_s(str, delim, saveptr);
}

#define _USE_MATH_DEFINES

#define srand48(x) srand(x)
#define drand48() (double(rand()) / RAND_MAX)

#include <sys/utime.h>
#define utime _utime;

#define RTLD_NOW 1
#define RTLD_LOCAL 1

/**
 * dl compatibility functions
 */

static inline void* dlopen(const char* filename, int flags) {
  return (void*)LoadLibrary(filename);
}

static inline int dlerror(void) {
  return GetLastError();
}

static inline void *dlsym(void* handle, const char *symbol) {
  return GetProcAddress((HMODULE)handle, symbol);
}

/**
 * Filesystem stuff
 */
#include <io.h>
#define R_OK    4       /* Test for read permission.  */
#define W_OK    2       /* Test for write permission.  */
#define F_OK    0       /* Test for existence.  */

#include <stdlib.h>
#include <direct.h>
#include <process.h>

#define mkdir(a,b) mkdir(a)

/**
 * TODO: test these implementations...
 */

/* Commented out: unused, and actually incorrect/unsafe.
static inline void gen_random(char *s, const int len) {
  static const char alphanum[] =
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz";

  for (int i = 0; i < len; ++i) {
    s[i] = alphanum[rand() % (sizeof(alphanum)-1)];
  }
  s[len] = 0;
}

static inline void mkdtemp(char *temp) {
  int rnd_start = strlen(temp) - 6;
  gen_random(&temp[rnd_start], 6);
  mkdir(temp);
}
*/

/**
 * Memory allocation functions
 */
#include <malloc.h>

static int posix_memalign(void **p, size_t align, size_t size) { 
   void *buf = _aligned_malloc(size, align);
   if (buf == NULL) return errno;
   *p = buf;
   return 0;
}

#define alloca _alloca

#endif
